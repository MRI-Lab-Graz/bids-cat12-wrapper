export function getRunOptions() {
  return {
    only: document.getElementById('optOnly').value.trim(),
    skip: document.getElementById('optSkip').value.trim(),
    from_step: document.getElementById('optFromStep').value.trim(),
    until_step: document.getElementById('optUntilStep').value.trim(),
    cat12_dir: document.getElementById('optCat12Dir').value.trim(),
    participants: document.getElementById('optParticipants').value.trim(),
    results_dir: document.getElementById('optResultsDir').value.trim(),
    modality: document.getElementById('optModality').value.trim(),
    use_matlab: document.getElementById('optUseMatlab').checked,
    force: document.getElementById('optForce').checked,
    dry_run: document.getElementById('optDryRun').checked
  };
}

export function setRunOptions(options = {}) {
  document.getElementById('optOnly').value = options.only || 'stats,report';
  document.getElementById('optSkip').value = options.skip || '';
  document.getElementById('optFromStep').value = options.from_step || '';
  document.getElementById('optUntilStep').value = options.until_step || '';
  document.getElementById('optCat12Dir').value = options.cat12_dir || '';
  document.getElementById('optParticipants').value = options.participants || '';
  document.getElementById('optResultsDir').value = options.results_dir || '';
  document.getElementById('optModality').value = options.modality || '';
  document.getElementById('optUseMatlab').checked = Boolean(options.use_matlab);
  document.getElementById('optForce').checked = Boolean(options.force);
  document.getElementById('optDryRun').checked = Boolean(options.dry_run);
}
