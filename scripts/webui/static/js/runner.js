export function getRunOptions() {
  return {
    mode: document.getElementById('optMode').value.trim(),
    cat12_dir: document.getElementById('optCat12Dir').value.trim(),
    participants: document.getElementById('optParticipants').value.trim(),
    stats_config: document.getElementById('optStatsConfig').value.trim(),
    modality: document.getElementById('optModality').value.trim(),
    force_all: document.getElementById('optForceAll').checked,
    dry_run: document.getElementById('optDryRun').checked,
    report_results_dir: document.getElementById('optReportResultsDir').value.trim(),
    report_quality: document.getElementById('optReportQuality').value.trim(),
    report_filter: document.getElementById('optReportFilter').value.trim(),
    report_output_html: document.getElementById('optReportOutputHtml').value.trim()
  };
}

export function setRunOptions(options = {}) {
  document.getElementById('optMode').value = options.mode || 'stats';
  document.getElementById('optCat12Dir').value = options.cat12_dir || '';
  document.getElementById('optParticipants').value = options.participants || '';
  document.getElementById('optStatsConfig').value = options.stats_config || '';
  document.getElementById('optModality').value = options.modality || '';
  document.getElementById('optForceAll').checked = Boolean(options.force_all);
  document.getElementById('optDryRun').checked = Boolean(options.dry_run);
  document.getElementById('optReportResultsDir').value = options.report_results_dir || '';
  document.getElementById('optReportQuality').value = options.report_quality || 'low';
  document.getElementById('optReportFilter').value = options.report_filter || 'no_tfce';
  document.getElementById('optReportOutputHtml').value = options.report_output_html || '';
}
