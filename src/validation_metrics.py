"""Explicit validation-only selection rules, independent of test outcomes."""
def selection_score(metrics,rule='group_macro'):
    groups=metrics.get('groups',metrics.get('evaluation'))
    if not groups:
        raise ValueError('Selection requires nonempty validation groups')
    if rule=='group_macro':
        return sum(g['exact_acc'] for g in groups.values())/len(groups)
    if rule=='balanced_real_synthetic':
        real=[g['exact_acc'] for name,g in groups.items() if name.startswith('public_')]
        synthetic=[g['exact_acc'] for name,g in groups.items() if not name.startswith('public_')]
        if not real or not synthetic:
            raise ValueError('Balanced selection requires public and synthetic validation')
        return .5*sum(real)/len(real)+.5*sum(synthetic)/len(synthetic)
    raise ValueError(f'Unknown selection rule: {rule}')
