"""Two-stage hybrid training with a matched CTC-only control and validation selection."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from .config import HybridConfig
from .data import RecognitionData, assert_isolated, read_manifest
from .evaluation import dump, evaluate
from .model import HybridRecognizer, load_hybrid
from .augmentation import TrainingAugment
from .provenance import capture_source


def save_checkpoint(model, path, **metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    torch.save(model.checkpoint(**metadata), temporary)
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--initial-checkpoint', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--ctc-control', action='store_true')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    try:
        capture_source(out/'configs/source_snapshot')
        settings = json.loads(Path(args.config).read_text())
        config = HybridConfig(**settings['model'])
        train_settings = settings['training']
        torch.set_num_threads(4)
        torch.manual_seed(train_settings['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('Training requires a CUDA-capable GPU')
        device = torch.device('cuda')
        initial_path = Path(args.initial_checkpoint)
        initial_sha = hashlib.sha256(initial_path.read_bytes()).hexdigest()
        source = torch.load(initial_path, map_location='cpu', weights_only=True)
        variant = settings.get('variant', 'baseline')
        if variant != 'baseline':
            from .experimental import ExperimentalRecognizer
            if not args.ctc_control:
                raise ValueError('This architecture sweep uses matched CTC-only training')
            model = ExperimentalRecognizer.from_parent(source, config, variant).to(device)
        else:
            model = HybridRecognizer.from_transfer(source, config).to(device)
        rows = {name: read_manifest(Path(args.data)/f'configs/{name}_manifest.json')
                for name in ['train', 'validation', 'evaluation']}
        assert_isolated(rows)
        for name, entries in rows.items():
            dump(out/f'configs/{name}_manifest.json', dict(samples=entries))
        dump(out/'configs/protocol.json', dict(settings=settings, ctc_control=args.ctc_control,
            initial_checkpoint=str(initial_path), initial_sha256=initial_sha,
            source_pretrained_sha256=source.get('pretrained_sha256'),
            selection='Same balanced real/synthetic validation metric; select weights and decoder without test data',
            matched_control='Same visual initialization, joint-update count, sample order, visual learning rates and per-step dropout seed',
            attention=('Continued from explicit trained hybrid checkpoint' if source.get('architecture')==model.architecture else
                       'Freshly initialized causal decoder; not PARSeq pretrained weights or a claimed paper reproduction'),
            attention_warmup_updates=0 if args.ctc_control else train_settings['warmup_steps'],
            joint_updates=train_settings['joint_steps'], repeats=1))
        dump(out/'logs/status.json', dict(state='preparing'))
        data = RecognitionData(rows['train'], model.vocabulary, config, device)
        validation = RecognitionData(rows['validation'], model.vocabulary, config, device)
        encoder = list(model.encoder.parameters())+list(model.foreground.parameters())
        ctc = list(model.ctc_neck.parameters())+list(model.ctc_head.parameters())
        attention = list(model.attention_parameters())
        variant_parameters = list(model.variant_block.parameters()) if hasattr(model, 'variant_block') else []
        optimizer = torch.optim.AdamW([dict(params=encoder, lr=train_settings['encoder_lr']),
            dict(params=ctc, lr=train_settings['ctc_lr']), dict(params=attention, lr=train_settings['attention_lr'])],
            weight_decay=train_settings['weight_decay'])
        if variant_parameters:
            optimizer.add_param_group(dict(params=variant_parameters, lr=train_settings['variant_lr']))
        criterion = torch.nn.CTCLoss(blank=model.vocabulary.blank, zero_infinity=False)
        amp = torch.cuda.is_bf16_supported()
        warmup = 0 if args.ctc_control else train_settings['warmup_steps']
        joint_steps = train_settings['joint_steps']
        augmentation = TrainingAugment(train_settings.get('augmentation_probability', 0.))
        best = dict(ctc=-1., attention=-1., hybrid=-1.)
        best_overall, selected_mode, best_step = -1., 'ctc', 0
        history = []
        metadata = dict(initial_sha256=initial_sha, training_samples=len(data.rows),
            pretrained_sha256=source.get('pretrained_sha256'),
            selection_rule='balanced_real_synthetic')

        def validate(step, phase, mode):
            nonlocal best_overall, selected_mode, best_step
            results = evaluate(model, validation, train_settings['evaluation_batch_size'], mode=mode)
            scores = {}
            for decoder, result in results.items():
                score = result['balanced_validation_score']
                if score is None:
                    raise ValueError('Validation requires both public and synthetic groups')
                scores[decoder] = score
                if score > best[decoder]:
                    best[decoder] = score
                    save_checkpoint(model, out/f'checkpoints/best_{decoder}.pth', **metadata,
                        attention_trained=not args.ctc_control and (phase!='initial' or source.get('attention_trained', False)),
                        selected_decoder=decoder, validation_score=score, step=step, phase=phase)
                if score > best_overall:
                    best_overall, selected_mode, best_step = score, decoder, step
                    save_checkpoint(model, out/'checkpoints/best_model.pth', **metadata,
                        attention_trained=not args.ctc_control and (phase!='initial' or source.get('attention_trained', False)),
                        selected_decoder=decoder, validation_score=score, step=step, phase=phase)
            dump(out/f'logs/validation_{phase}_{step}.json', results)
            print('VALIDATION', phase, step, scores, flush=True)
            return scores

        validate(0, 'initial', 'ctc')
        start = time.monotonic()
        for phase, steps in [('warmup', warmup), ('joint', joint_steps)]:
            if not steps:
                continue
            for parameter in encoder+ctc:
                parameter.requires_grad_(phase == 'joint')
            for parameter in attention:
                parameter.requires_grad_(not args.ctc_control)
            generator = torch.Generator(device=device).manual_seed(train_settings['seed'] + (10000 if phase=='warmup' else 20000))
            augmentation_generator = torch.Generator(device=device).manual_seed(train_settings['seed'] + (50000 if phase=='warmup' else 60000))
            for step in range(1, steps+1):
                torch.manual_seed(train_settings['seed'] + step + (30000 if phase=='warmup' else 40000))
                model.train()
                model.freeze_statistics()
                if phase == 'warmup':
                    rates = [0., 0., train_settings['attention_lr']*min(1., step/25)]
                else:
                    scale = .01 + .99*.5*(1+math.cos(math.pi*(step-1)/max(1, steps-1)))
                    rates = [train_settings['encoder_lr']*scale, train_settings['ctc_lr']*scale,
                             train_settings['attention_joint_lr']*scale]
                    if variant_parameters:
                        rates.append(train_settings['variant_lr']*scale)
                for group, rate in zip(optimizer.param_groups, rates):
                    group['lr'] = rate
                selected, _ = data.sample(train_settings['batch_size'], generator)
                batch_images, batch_masks = augmentation(data.images[selected], data.masks[selected], data.widths[selected], augmentation_generator)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    if args.ctc_control:
                        result = model.encode(batch_images, data.widths[selected])
                    else:
                        result = model.forward_training(batch_images, data.widths[selected], data.tokens[selected])
                target, lengths = data.ctc_target(selected)
                ctc_loss = criterion(result['ctc'].float().log_softmax(-1).transpose(0, 1), target,
                    torch.full((len(selected),), result['ctc'].shape[1]), lengths)
                ar_loss = torch.zeros((), device=device)
                if not args.ctc_control:
                    ar_loss = torch.nn.functional.cross_entropy(result['attention'].float().transpose(1, 2),
                        data.tokens[selected, 1:], ignore_index=model.vocabulary.pad, label_smoothing=.05)
                loss = ar_loss if phase=='warmup' else ctc_loss + train_settings['attention_weight']*ar_loss
                annotated = data.annotated[selected]
                if phase=='joint' and annotated.any():
                    mask = torch.nn.functional.adaptive_avg_pool2d(batch_masks[annotated].float(), result['foreground'].shape[-2:])
                    loss = loss + .2*torch.nn.functional.binary_cross_entropy_with_logits(
                        result['foreground'][annotated].float(), mask, pos_weight=torch.tensor(3., device=device))
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training objective')
                loss.backward()
                if step == 1:
                    gradients = {name: sum(float(p.grad.abs().sum()) for p in parameters if p.grad is not None)
                                 for name, parameters in [('encoder', encoder), ('ctc', ctc), ('attention', attention)]}
                    if (phase=='joint' and gradients['encoder'] <= 0) or (not args.ctc_control and gradients['attention'] <= 0):
                        raise RuntimeError('Required training path is detached')
                    if phase=='warmup' and (gradients['encoder'] or gradients['ctc']):
                        raise RuntimeError('Frozen visual path received warmup gradients')
                    if variant_parameters:
                        gradients['variant'] = sum(float(p.grad.abs().sum()) for p in variant_parameters if p.grad is not None)
                        if gradients['variant'] <= 0:
                            raise RuntimeError('Experimental branch received no gradient')
                    dump(out/f'logs/gradients_{phase}.json', gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                if step % 50 == 0:
                    dump(out/'logs/status.json', dict(state='training', phase=phase, step=step, steps=steps))
                    print('TRAIN', phase, step, round(float(ctc_loss), 4), round(float(ar_loss), 4), flush=True)
                if step % train_settings['evaluate_every'] == 0 or step == steps:
                    scores = validate(step, phase, 'ctc' if args.ctc_control else 'all')
                    history.append(dict(phase=phase, step=step, ctc_loss=float(ctc_loss), attention_loss=float(ar_loss), scores=scores))
                    dump(out/'logs/history.json', history)
        if joint_steps:
            save_checkpoint(model, out/'checkpoints/last_ctc.pth', **metadata,
                attention_trained=not args.ctc_control, selected_decoder='ctc',
                validation_score=scores['ctc'], step=joint_steps, phase='joint',
                checkpoint_role='Last trained state for diagnostics; not selected over validation best')
        dump(out/'logs/results.json', dict(selected_decoder=selected_mode, best_step=best_step,
            validation_score=best_overall, decoder_best_scores=best, seconds=time.monotonic()-start,
            production_ready=False, test_evaluated_during_training=False))
        dump(out/'logs/status.json', dict(state='complete', selected_decoder=selected_mode))
    except Exception as exc:
        dump(out/'logs/status.json', dict(state='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
