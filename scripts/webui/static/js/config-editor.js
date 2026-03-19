function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPrimitive(value) {
  return value === null || ['string', 'number', 'boolean'].includes(typeof value);
}

function isCommentKey(key) {
  if (!key) return false;
  return key === '_updated' || key.startsWith('_') || key.endsWith('_comment');
}

function prettifyKey(key) {
  if (!key) return '';
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (s) => s.toUpperCase());
}

const SECTION_META = {
  study: { title: 'Study', subtitle: 'Project identity and description' },
  paths: { title: 'Paths', subtitle: 'Project-relative input/output paths' },
  preprocessing: { title: 'Preprocessing', subtitle: 'BIDS source, stages, smoothing, and execution' },
  statistics: { title: 'Statistics', subtitle: 'Multimodality design, inference, and reporting' },
  software: { title: 'Software Mode', subtitle: 'Execution backend and standalone paths' },
  matlab: { title: 'MATLAB', subtitle: 'MATLAB executable and rendering behavior' },
  spm: { title: 'SPM', subtitle: 'SPM installation path' },
  python: { title: 'Python', subtitle: 'Python executable configuration' },
  analysis: { title: 'Analysis Setup', subtitle: 'Participants, groups, sessions, modalities' },
  screening: { title: 'Screening', subtitle: 'Initial significance and cluster filters' },
  tfce: { title: 'TFCE', subtitle: 'Permutation-based correction settings' },
  double_threshold: { title: 'Double Threshold', subtitle: 'SPM intensity + cluster thresholding' },
  reporting: { title: 'Reporting', subtitle: 'Output report and p-value labels' },
  performance: { title: 'Performance', subtitle: 'Parallel jobs and memory limits' },
  output: { title: 'Output', subtitle: 'Naming and cleanup behavior' },
  pipeline: { title: 'Pipeline Steps', subtitle: 'Enable or disable major pipeline stages' },
};

const SECTION_GROUPS = [
  {
    title: 'Pipeline Setup',
    subtitle: 'Study context plus preprocessing and statistics definitions',
    keys: ['study', 'paths', 'preprocessing', 'statistics', 'analysis'],
  },
  {
    title: 'Technical & Software',
    subtitle: 'Runtime mode, executables, software paths, and performance',
    keys: ['software', 'matlab', 'spm', 'python', 'performance'],
  },
  {
    title: 'Statistics & Reporting',
    subtitle: 'Inference thresholds, TFCE, double-threshold, and reports',
    keys: ['screening', 'tfce', 'double_threshold', 'reporting'],
  },
  {
    title: 'Project & Output',
    subtitle: 'Study metadata, pipeline toggles, and output naming',
    keys: ['study', 'pipeline', 'output'],
  },
];

const ENUM_OPTIONS = {
  'software.mode': ['matlab', 'standalone'],
  'reporting.quality': ['low', 'standard', 'publication'],
  'reporting.filter': ['all', 'tfce', 'no_tfce', 'spmt', 'double_threshold'],
};

function isModalityNamePath(tokens) {
  const isLegacy =
    tokens.length === 4 &&
    tokens[0] === 'analysis' &&
    tokens[1] === 'modalities' &&
    typeof tokens[2] === 'number' &&
    tokens[3] === 'name';

  const isUnified =
    tokens.length === 5 &&
    tokens[0] === 'statistics' &&
    tokens[1] === 'design' &&
    tokens[2] === 'modalities' &&
    typeof tokens[3] === 'number' &&
    tokens[4] === 'name';

  return isLegacy || isUnified;
}

function isCovariatesPath(tokens) {
  const isLegacy =
    tokens.length === 4 &&
    tokens[0] === 'analysis' &&
    tokens[1] === 'modalities' &&
    typeof tokens[2] === 'number' &&
    tokens[3] === 'covariates';

  const isUnified =
    tokens.length === 5 &&
    tokens[0] === 'statistics' &&
    tokens[1] === 'design' &&
    tokens[2] === 'modalities' &&
    typeof tokens[3] === 'number' &&
    tokens[4] === 'covariates';

  return isLegacy || isUnified;
}

function isGroupColumnPath(tokens) {
  const isLegacy = tokens.length === 2 && tokens[0] === 'analysis' && tokens[1] === 'group_column';
  const isUnified = tokens.length === 3 && tokens[0] === 'statistics' && tokens[1] === 'input' && tokens[2] === 'group_column';
  return isLegacy || isUnified;
}

function isModalityMaskPath(tokens) {
  const isLegacy =
    tokens.length === 4 &&
    tokens[0] === 'analysis' &&
    tokens[1] === 'modalities' &&
    typeof tokens[2] === 'number' &&
    tokens[3] === 'mask';

  const isUnified =
    tokens.length === 5 &&
    tokens[0] === 'statistics' &&
    tokens[1] === 'design' &&
    tokens[2] === 'modalities' &&
    typeof tokens[3] === 'number' &&
    tokens[4] === 'mask';

  return isLegacy || isUnified;
}

function isModalityFolderPath(tokens) {
  const isLegacy =
    tokens.length === 4 &&
    tokens[0] === 'analysis' &&
    tokens[1] === 'modalities' &&
    typeof tokens[2] === 'number' &&
    tokens[3] === 'folder_name';

  const isUnified =
    tokens.length === 5 &&
    tokens[0] === 'statistics' &&
    tokens[1] === 'design' &&
    tokens[2] === 'modalities' &&
    typeof tokens[3] === 'number' &&
    tokens[4] === 'folder_name';

  return isLegacy || isUnified;
}

function isPreprocessingBidsDirPath(tokens) {
  return (
    tokens.length === 3 &&
    tokens[0] === 'preprocessing' &&
    tokens[1] === 'bids' &&
    tokens[2] === 'bids_dir'
  );
}

function isPreprocessingOpenNeuroDatasetPath(tokens) {
  return (
    tokens.length === 3 &&
    tokens[0] === 'preprocessing' &&
    tokens[1] === 'bids' &&
    tokens[2] === 'openneuro_dataset'
  );
}

function isPreprocessingOpenNeuroFlagPath(tokens) {
  return (
    tokens.length === 3 &&
    tokens[0] === 'preprocessing' &&
    tokens[1] === 'bids' &&
    ['openneuro', 'openneuro_download', 'openneuro_download_only_anat', 'openneuro_download_all'].includes(tokens[2])
  );
}

function isProjectFolderPath(tokens) {
  return tokens.length === 2 && tokens[0] === 'study' && tokens[1] === 'project_folder';
}

function isManagedFolderPath(tokens) {
  if (isProjectFolderPath(tokens)) return true;
  return tokens.length >= 2 && tokens[0] === 'paths';
}

function normalizeFolderToken(value, fallback = 'item') {
  const token = String(value ?? '')
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '-')
    .replace(/[^a-z0-9-_]/g, '');
  return token || fallback;
}

function formatSmoothingToken(smoothing) {
  if (smoothing === null || smoothing === undefined || String(smoothing).trim() === '') {
    return 'auto';
  }

  const numeric = Number(smoothing);
  if (!Number.isNaN(numeric)) {
    return `${numeric}mm`;
  }

  const raw = normalizeFolderToken(String(smoothing), 'auto');
  return raw.endsWith('mm') ? raw : `${raw}mm`;
}

function buildAutoModalityFolderName(modality) {
  const modalityName = normalizeFolderToken(modality?.name, 'modality');
  const covariates = Array.isArray(modality?.covariates)
    ? modality.covariates.map((item) => normalizeFolderToken(item)).filter(Boolean)
    : [];
  const covToken = covariates.length ? covariates.join('-') : 'nocov';
  const smoothingToken = formatSmoothingToken(modality?.smoothing_kernel);
  return `${modalityName}_${covToken}_${smoothingToken}`;
}

function stripCommentKeysDeep(value) {
  if (Array.isArray(value)) {
    return value.map(stripCommentKeysDeep);
  }
  if (isPlainObject(value)) {
    const clean = {};
    Object.entries(value).forEach(([k, v]) => {
      if (isCommentKey(k)) return;
      clean[k] = stripCommentKeysDeep(v);
    });
    return clean;
  }
  return value;
}

function getByPath(obj, tokens) {
  return tokens.reduce((acc, token) => acc?.[token], obj);
}

function setByPath(obj, tokens, value) {
  let cursor = obj;
  for (let i = 0; i < tokens.length - 1; i++) {
    cursor = cursor[tokens[i]];
  }
  cursor[tokens[tokens.length - 1]] = value;
}

function parseValue(raw, original) {
  if (typeof original === 'boolean') return raw === 'true';
  if (typeof original === 'number') {
    if (raw === '') return original;
    return Number.isInteger(original) ? parseInt(raw, 10) : parseFloat(raw);
  }
  if (original === null) return raw === '' ? null : raw;
  return raw;
}

function parsePrimitiveArray(raw, original) {
  const trimmed = raw.trim();
  if (!trimmed) return [];

  const items = trimmed
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean);

  const sample = Array.isArray(original) && original.length ? original[0] : '';
  if (typeof sample === 'number') {
    return items.map((item) => {
      const parsed = Number(item);
      return Number.isNaN(parsed) ? item : parsed;
    });
  }
  if (typeof sample === 'boolean') {
    return items.map((item) => item.toLowerCase() === 'true');
  }
  return items;
}

export class ConfigEditor {
  constructor(rootEl, badgeEl) {
    this.rootEl = rootEl;
    this.badgeEl = badgeEl;
    this.source = {};
    this.fieldCount = 0;
    this.participantsColumns = [];
  }

  setParticipantsColumns(columns = []) {
    const draft = this.collect();
    this.participantsColumns = Array.from(new Set(columns.filter(Boolean)));
    this.load(draft);
  }

  load(configData) {
    const openState = this.captureOpenState();
    this.source = stripCommentKeysDeep(configData || {});
    this.fieldCount = 0;
    this.rootEl.innerHTML = '';

    const topKeys = Object.keys(this.source).filter((key) => !isCommentKey(key));

    const groups = this.buildSectionGroups(topKeys);
    groups.forEach((group) => {
      const section = this.createSection(group.title, group.subtitle, false, `section:${group.title}`);
      this.rootEl.appendChild(section.wrapper);

      const singleKey = group.keys.length === 1;
      group.keys.forEach((key) => {
        this.renderTopLevelKey(section.content, key, singleKey);
      });
    });

    this.badgeEl.textContent = `${this.fieldCount} fields`;
    this.applyOpenState(openState);
  }

  captureOpenState() {
    const state = {};
    this.rootEl.querySelectorAll('details[data-state-key]').forEach((node) => {
      state[node.dataset.stateKey] = node.open;
    });
    return state;
  }

  applyOpenState(state = {}) {
    this.rootEl.querySelectorAll('details[data-state-key]').forEach((node) => {
      const key = node.dataset.stateKey;
      if (Object.prototype.hasOwnProperty.call(state, key)) {
        node.open = Boolean(state[key]);
      }
    });
  }

  buildSectionGroups(topKeys) {
    const present = new Set(topKeys);
    const consumed = new Set();
    const groups = [];

    SECTION_GROUPS.forEach((group) => {
      const keys = group.keys.filter((key) => present.has(key));
      if (!keys.length) return;

      keys.forEach((key) => consumed.add(key));
      groups.push({
        title: group.title,
        subtitle: group.subtitle,
        keys,
      });
    });

    const remaining = topKeys.filter((key) => !consumed.has(key));
    if (remaining.length) {
      groups.push({
        title: 'Additional Settings',
        subtitle: 'Other configuration fields',
        keys: remaining,
      });
    }

    return groups;
  }

  renderTopLevelKey(container, key, singleKeyInGroup = false) {
    const value = this.source[key];
    if (singleKeyInGroup) {
      this.renderNode(container, value, [key], 0);
      return;
    }

    const meta = SECTION_META[key] || {
      title: prettifyKey(key),
      subtitle: 'Configuration settings',
    };

    const block = this.createCardBlock(meta.title, false, `top:${key}`);
    container.appendChild(block.wrapper);

    if (meta.subtitle) {
      const subtitle = document.createElement('div');
      subtitle.className = 'ux-card-subtitle';
      subtitle.textContent = meta.subtitle;
      block.content.appendChild(subtitle);
    }

    this.renderNode(block.content, value, [key], 0);
  }

  getValueAtPath(tokens) {
    const snapshot = this.collect();
    return getByPath(snapshot, tokens);
  }

  setValueAtPath(tokens, value) {
    const updated = this.collect();
    setByPath(updated, tokens, value);
    this.load(updated);
  }

  renderNode(container, value, tokens, depth = 0) {
    if (isPrimitive(value)) {
      const key = typeof tokens[tokens.length - 1] === 'number' ? `[${tokens[tokens.length - 1]}]` : String(tokens[tokens.length - 1]);
      this.renderLeaf(container, key, value, tokens);
      return;
    }

    if (Array.isArray(value)) {
      if (isCovariatesPath(tokens) && value.every(isPrimitive)) {
        this.renderCovariatesSelector(container, tokens, value.map(v => String(v)));
        return;
      }

      if (value.every(isPrimitive)) {
        this.renderPrimitiveArrayField(container, tokens, value);
        return;
      }

      const listWrap = document.createElement('div');
      listWrap.className = 'ux-list-wrap';
      container.appendChild(listWrap);

      value.forEach((item, index) => {
        const childTokens = [...tokens, index];
        if (isPrimitive(item)) {
          this.renderLeaf(listWrap, `[${index + 1}]`, item, childTokens);
        } else {
          const groupKey = String(tokens[tokens.length - 1]);
          if (groupKey === 'modalities') {
            this.renderNode(listWrap, item, childTokens, depth + 1);
          } else {
            const itemTitle = `${prettifyKey(groupKey)} ${index + 1}`;
            const block = this.createCardBlock(itemTitle, false, `path:${childTokens.join('.')}`);
            listWrap.appendChild(block.wrapper);
            this.renderNode(block.content, item, childTokens, depth + 1);
          }
        }
      });
      return;
    }

    if (isPlainObject(value)) {
      const entries = Object.entries(value).filter(([key]) => !isCommentKey(key));

      if (entries.length === 0) return;

      const shouldWrapAsCard = depth > 0 && typeof tokens[tokens.length - 1] !== 'number';
      const contentRoot = shouldWrapAsCard
        ? (() => {
            const key = String(tokens[tokens.length - 1]);
            const block = this.createCardBlock(prettifyKey(key), false, `path:${tokens.join('.')}`);
            container.appendChild(block.wrapper);
            return block.content;
          })()
        : container;

      const group = document.createElement('div');
      group.className = 'ux-fields-grid';
      contentRoot.appendChild(group);

      Object.entries(value).forEach(([key, childValue]) => {
        if (isCommentKey(key)) return;
        if (isPlainObject(childValue) || Array.isArray(childValue)) {
          const childTokens = [...tokens, key];
          this.renderNode(group, childValue, childTokens, depth + 1);
        } else {
          this.renderLeaf(group, key, childValue, [...tokens, key]);
        }
      });
      return;
    }
  }

  createSection(titleText, subtitleText, open = false, stateKey = '') {
    const wrapper = document.createElement('details');
    wrapper.className = 'ux-section';
    wrapper.open = open;
    if (stateKey) {
      wrapper.dataset.stateKey = stateKey;
    }

    const summary = document.createElement('summary');
    summary.className = 'ux-section-summary';

    const title = document.createElement('div');
    title.className = 'ux-section-title';
    title.textContent = titleText;

    const subtitle = document.createElement('div');
    subtitle.className = 'ux-section-subtitle';
    subtitle.textContent = subtitleText;

    summary.appendChild(title);
    summary.appendChild(subtitle);

    const content = document.createElement('div');
    content.className = 'ux-section-content';

    wrapper.appendChild(summary);
    wrapper.appendChild(content);

    return { wrapper, content };
  }

  createCardBlock(titleText, open = true, stateKey = '') {
    const wrapper = document.createElement('details');
    wrapper.className = 'ux-card';
    wrapper.open = open;
    if (stateKey) {
      wrapper.dataset.stateKey = stateKey;
    }

    const summary = document.createElement('summary');
    summary.className = 'ux-card-title';
    summary.textContent = titleText;

    const content = document.createElement('div');
    content.className = 'ux-card-content';

    wrapper.appendChild(summary);
    wrapper.appendChild(content);

    return { wrapper, content };
  }

  renderLeaf(container, key, value, tokens) {
    this.fieldCount += 1;

    const row = document.createElement('div');
    row.className = 'ux-field';

    const keyName = String(tokens[tokens.length - 1]);
    const path = tokens.join('.');

    const label = document.createElement('label');
    label.className = 'form-label mb-1 ux-label';
    label.textContent = prettifyKey(String(key));

    let input;
    if (isPreprocessingBidsDirPath(tokens)) {
      const group = document.createElement('div');
      group.className = 'input-group input-group-sm';

      const bidsTokens = tokens.slice(0, -1);
      const openNeuroTokens = [...bidsTokens, 'openneuro_dataset'];
      const openNeuroEnabled = Boolean(getByPath(this.source, [...bidsTokens, 'openneuro']));
      const openNeuroDataset = getByPath(this.source, openNeuroTokens);

      input = document.createElement('input');
      input.className = 'form-control';
      input.type = 'text';
      input.placeholder = openNeuroEnabled ? 'ds004937' : 'path/to/bids_folder';
      input.value = String(openNeuroEnabled ? (openNeuroDataset || value || '') : (value || ''));
      input.dataset.bidsSourceField = 'true';
      input.dataset.sourceMode = openNeuroEnabled ? 'openneuro' : 'local';
      input.dataset.path = JSON.stringify(tokens);

      const localBtn = document.createElement('button');
      localBtn.type = 'button';
      localBtn.className = `btn ${openNeuroEnabled ? 'btn-outline-secondary' : 'btn-secondary'}`;
      localBtn.textContent = 'Local BIDS';
      localBtn.dataset.bidsModeLocal = 'true';
      localBtn.dataset.path = JSON.stringify(bidsTokens);

      const openNeuroBtn = document.createElement('button');
      openNeuroBtn.type = 'button';
      openNeuroBtn.className = `btn ${openNeuroEnabled ? 'btn-secondary' : 'btn-outline-secondary'}`;
      openNeuroBtn.textContent = 'OpenNeuro';
      openNeuroBtn.dataset.openneuroActivate = 'true';
      openNeuroBtn.dataset.path = JSON.stringify(bidsTokens);

      label.textContent = openNeuroEnabled ? 'OpenNeuro Dataset ID' : 'BIDS Folder';
      group.appendChild(input);
      group.appendChild(localBtn);
      group.appendChild(openNeuroBtn);

      row.appendChild(label);
      row.appendChild(group);
      container.appendChild(row);
      return;
    }

    if (ENUM_OPTIONS[path]) {
      input = document.createElement('select');
      input.className = 'form-select form-select-sm';
      ENUM_OPTIONS[path].forEach((mode) => {
        const opt = document.createElement('option');
        opt.value = mode;
        opt.textContent = mode;
        input.appendChild(opt);
      });
      input.value = String(value ?? ENUM_OPTIONS[path][0]);
    } else if (isGroupColumnPath(tokens)) {
      input = document.createElement('select');
      input.className = 'form-select form-select-sm';
      const options = Array.from(new Set([...this.participantsColumns, String(value || '')].filter(Boolean)));
      options.forEach((columnName) => {
        const opt = document.createElement('option');
        opt.value = columnName;
        opt.textContent = columnName;
        input.appendChild(opt);
      });
      if (!options.length) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = 'Select participants file first';
        input.appendChild(placeholder);
      }
      input.value = String(value || '');
    } else if (isModalityMaskPath(tokens)) {
      const group = document.createElement('div');
      group.className = 'input-group input-group-sm';

      input = document.createElement('input');
      input.className = 'form-control';
      input.type = 'text';
      input.placeholder = 'No mask';
      input.value = value === null ? '' : String(value);

      const browseBtn = document.createElement('button');
      browseBtn.type = 'button';
      browseBtn.className = 'btn btn-outline-secondary';
      browseBtn.textContent = 'Browse...';
      browseBtn.dataset.maskPicker = 'true';
      browseBtn.dataset.path = JSON.stringify(tokens);

      group.appendChild(input);
      group.appendChild(browseBtn);

      input.dataset.configField = 'true';
      input.dataset.path = JSON.stringify(tokens);

      row.appendChild(label);
      if (keyName === 'participants_file' || isModalityFolderPath(tokens)) {
        row.classList.add('ux-hidden-field');
      }
      row.appendChild(group);
      container.appendChild(row);
      return;
    } else if (isModalityNamePath(tokens)) {
      input = document.createElement('select');
      input.className = 'form-select form-select-sm';
      Array.from(new Set(['vbm', 'thickness', 'depth', 'gyrification', String(value || '')].filter(Boolean))).forEach((mode) => {
        const opt = document.createElement('option');
        opt.value = mode;
        opt.textContent = mode;
        input.appendChild(opt);
      });
      input.value = String(value || 'vbm');
    } else if (isManagedFolderPath(tokens)) {
      const group = document.createElement('div');
      group.className = 'input-group input-group-sm';

      input = document.createElement('input');
      input.className = 'form-control';
      input.type = 'text';
      input.value = value === null ? '' : String(value);

      const browseBtn = document.createElement('button');
      browseBtn.type = 'button';
      browseBtn.className = 'btn btn-outline-secondary';
      browseBtn.textContent = 'Browse...';
      browseBtn.dataset.folderPicker = 'true';
      browseBtn.dataset.folderPickerTitle = isProjectFolderPath(tokens) ? 'Pick Project Folder' : `Pick ${prettifyKey(String(tokens[tokens.length - 1]))}`;
      browseBtn.dataset.path = JSON.stringify(tokens);

      const clearBtn = document.createElement('button');
      clearBtn.type = 'button';
      clearBtn.className = 'btn btn-outline-danger';
      clearBtn.textContent = 'X';
      clearBtn.title = 'Clear value';
      clearBtn.dataset.clearField = 'true';
      clearBtn.dataset.path = JSON.stringify(tokens);

      input.dataset.configField = 'true';
      input.dataset.path = JSON.stringify(tokens);

      group.appendChild(input);
      group.appendChild(browseBtn);
      group.appendChild(clearBtn);

      row.appendChild(label);
      row.appendChild(group);
      container.appendChild(row);
      return;
    } else if (typeof value === 'boolean') {
      input = document.createElement('select');
      input.className = 'form-select form-select-sm';
      ['true', 'false'].forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        opt.textContent = v;
        input.appendChild(opt);
      });
      input.value = String(value);
    } else {
      input = document.createElement('input');
      input.className = 'form-control form-control-sm';
      input.type = typeof value === 'number' ? 'number' : 'text';
      input.value = value === null ? '' : String(value);
    }

    input.dataset.configField = 'true';
    input.dataset.path = JSON.stringify(tokens);

    row.appendChild(label);
    if (
      keyName === 'participants_file' ||
      isModalityFolderPath(tokens) ||
      isPreprocessingOpenNeuroDatasetPath(tokens) ||
      isPreprocessingOpenNeuroFlagPath(tokens)
    ) {
      row.classList.add('ux-hidden-field');
    }
    row.appendChild(input);
    container.appendChild(row);
  }

  renderPrimitiveArrayField(container, tokens, values) {
    this.fieldCount += 1;

    const row = document.createElement('div');
    row.className = 'ux-field ux-field-wide';

    const label = document.createElement('label');
    label.className = 'form-label mb-1 ux-label';
    label.textContent = prettifyKey(String(tokens[tokens.length - 1]));

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-control form-control-sm';
    input.placeholder = 'Comma-separated values';
    input.value = values.map((v) => String(v)).join(', ');
    input.dataset.arrayPrimitiveField = 'true';
    input.dataset.path = JSON.stringify(tokens);

    row.appendChild(label);
    row.appendChild(input);
    container.appendChild(row);
  }

  renderCovariatesSelector(container, tokens, currentValues) {
    this.fieldCount += 1;

    const row = document.createElement('div');
    row.className = 'ux-field ux-field-wide';

    const label = document.createElement('label');
    label.className = 'form-label mb-2 ux-label';
    label.textContent = 'Covariates';

    const box = document.createElement('div');
    box.className = 'ux-checklist';
    box.dataset.covariatesField = 'true';
    box.dataset.path = JSON.stringify(tokens);

    const options = Array.from(
      new Set(['tiv', ...this.participantsColumns, ...currentValues.filter((v) => v === 'tiv' || this.participantsColumns.includes(v))].filter(Boolean))
    );

    options.forEach((item) => {
      const safeId = `cov_${tokens.join('_')}_${item}`.replace(/[^a-zA-Z0-9_]/g, '_');
      const wrap = document.createElement('div');
      wrap.className = 'form-check';

      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'form-check-input';
      checkbox.id = safeId;
      checkbox.value = item;
      checkbox.checked = currentValues.includes(item);

      const lbl = document.createElement('label');
      lbl.className = 'form-check-label';
      lbl.setAttribute('for', safeId);
      lbl.textContent = item;

      wrap.appendChild(checkbox);
      wrap.appendChild(lbl);
      box.appendChild(wrap);
    });

    row.appendChild(label);
    row.appendChild(box);
    container.appendChild(row);
  }

  collect() {
    const updated = structuredClone(this.source);

    const bidsSourceFields = this.rootEl.querySelectorAll('[data-bids-source-field="true"]');
    bidsSourceFields.forEach((el) => {
      const tokens = JSON.parse(el.dataset.path);
      const enteredValue = String(el.value || '').trim();
      const bidsTokens = tokens.slice(0, -1);
      const sourceMode = el.dataset.sourceMode === 'openneuro' ? 'openneuro' : 'local';

      if (sourceMode === 'openneuro') {
        const datasetId = enteredValue.toLowerCase();
        setByPath(updated, tokens, datasetId);
        setByPath(updated, [...bidsTokens, 'openneuro'], true);
        setByPath(updated, [...bidsTokens, 'openneuro_dataset'], datasetId);
        setByPath(updated, [...bidsTokens, 'openneuro_download'], true);
        setByPath(updated, [...bidsTokens, 'openneuro_download_only_anat'], true);
        setByPath(updated, [...bidsTokens, 'openneuro_download_all'], false);
        if (datasetId) {
          setByPath(updated, [...bidsTokens, 'source'], `OpenNeuro ${datasetId}`);
        }
      } else {
        setByPath(updated, tokens, enteredValue);
        setByPath(updated, [...bidsTokens, 'openneuro'], false);
        setByPath(updated, [...bidsTokens, 'openneuro_download'], false);
      }
    });

    const fields = this.rootEl.querySelectorAll('[data-config-field="true"]');
    fields.forEach((el) => {
      const tokens = JSON.parse(el.dataset.path);
      const original = getByPath(this.source, tokens);
      const parsed = parseValue(el.value, original);
      setByPath(updated, tokens, parsed);
    });

    const primitiveArrayFields = this.rootEl.querySelectorAll('[data-array-primitive-field="true"]');
    primitiveArrayFields.forEach((el) => {
      const tokens = JSON.parse(el.dataset.path);
      const original = getByPath(this.source, tokens);
      const parsed = parsePrimitiveArray(el.value, original);
      setByPath(updated, tokens, parsed);
    });

    const covariateFields = this.rootEl.querySelectorAll('[data-covariates-field="true"]');
    covariateFields.forEach((el) => {
      const tokens = JSON.parse(el.dataset.path);
      const selected = Array.from(el.querySelectorAll('input[type="checkbox"]:checked')).map((item) => item.value);
      setByPath(updated, tokens, selected);
    });

    const legacyModalities = updated?.analysis?.modalities;
    if (Array.isArray(legacyModalities)) {
      legacyModalities.forEach((modality) => {
        if (isPlainObject(modality)) {
          modality.folder_name = buildAutoModalityFolderName(modality);
        }
      });
    }

    const unifiedModalities = updated?.statistics?.design?.modalities;
    if (Array.isArray(unifiedModalities)) {
      unifiedModalities.forEach((modality) => {
        if (isPlainObject(modality)) {
          modality.folder_name = buildAutoModalityFolderName(modality);
        }
      });
    }

    return stripCommentKeysDeep(updated);
  }
}
