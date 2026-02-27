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

function isModalityNamePath(tokens) {
  return (
    tokens.length === 4 &&
    tokens[0] === 'analysis' &&
    tokens[1] === 'modalities' &&
    typeof tokens[2] === 'number' &&
    tokens[3] === 'name'
  );
}

function isCovariatesPath(tokens) {
  return (
    tokens.length === 4 &&
    tokens[0] === 'analysis' &&
    tokens[1] === 'modalities' &&
    typeof tokens[2] === 'number' &&
    tokens[3] === 'covariates'
  );
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
    this.source = stripCommentKeysDeep(configData || {});
    this.fieldCount = 0;
    this.rootEl.innerHTML = '';
    this.renderNode(this.rootEl, this.source, []);
    this.badgeEl.textContent = `${this.fieldCount} fields`;
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

  renderNode(container, value, tokens) {
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

      value.forEach((item, index) => {
        const childTokens = [...tokens, index];
        if (isPrimitive(item)) {
          this.renderLeaf(container, `[${index}]`, item, childTokens);
        } else {
          const block = this.createCollapsibleBlock(`[${index}]`, childTokens, tokens.length <= 1);
          container.appendChild(block.wrapper);
          this.renderNode(block.content, item, childTokens);
        }
      });
      return;
    }

    if (isPlainObject(value)) {
      Object.entries(value).forEach(([key, childValue]) => {
        if (isCommentKey(key)) return;
        if (isPlainObject(childValue) || Array.isArray(childValue)) {
          const childTokens = [...tokens, key];
          const block = this.createCollapsibleBlock(key, childTokens, tokens.length <= 1);
          container.appendChild(block.wrapper);
          this.renderNode(block.content, childValue, childTokens);
        } else {
          this.renderLeaf(container, key, childValue, [...tokens, key]);
        }
      });
      return;
    }
  }

  createCollapsibleBlock(titleText, tokens, open = false) {
    const wrapper = document.createElement('details');
    wrapper.className = 'field-block field-block-collapsible';
    wrapper.open = open;

    const summary = document.createElement('summary');
    summary.className = 'field-title';
    summary.textContent = titleText;

    const hint = document.createElement('div');
    hint.className = 'path-hint mb-1';
    hint.textContent = tokens.join('.');

    const content = document.createElement('div');
    content.className = 'field-block-content';

    wrapper.appendChild(summary);
    wrapper.appendChild(hint);
    wrapper.appendChild(content);

    return { wrapper, content };
  }

  renderLeaf(container, key, value, tokens) {
    this.fieldCount += 1;

    const row = document.createElement('div');
    row.className = 'field-leaf';

    const label = document.createElement('label');
    label.className = 'form-label mb-1';
    label.textContent = key;

    const hint = document.createElement('div');
    hint.className = 'path-hint mb-1';
    hint.textContent = tokens.join('.');

    let input;
    if (isModalityNamePath(tokens)) {
      input = document.createElement('select');
      input.className = 'form-select form-select-sm';
      ['vbm', 'thickness'].forEach((mode) => {
        const opt = document.createElement('option');
        opt.value = mode;
        opt.textContent = mode;
        input.appendChild(opt);
      });
      input.value = String(value || 'vbm');
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
    row.appendChild(hint);
    row.appendChild(input);
    container.appendChild(row);
  }

  renderCovariatesSelector(container, tokens, currentValues) {
    this.fieldCount += 1;

    const row = document.createElement('div');
    row.className = 'field-leaf';

    const label = document.createElement('label');
    label.className = 'form-label mb-1';
    label.textContent = 'covariates';

    const hint = document.createElement('div');
    hint.className = 'path-hint mb-2';
    hint.textContent = `${tokens.join('.')} (TIV + participants columns)`;

    const box = document.createElement('div');
    box.className = 'd-flex flex-wrap gap-2';
    box.dataset.covariatesField = 'true';
    box.dataset.path = JSON.stringify(tokens);

    const options = Array.from(
      new Set(['tiv', ...this.participantsColumns, ...currentValues].filter(Boolean))
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
    row.appendChild(hint);
    row.appendChild(box);
    container.appendChild(row);
  }

  collect() {
    const updated = structuredClone(this.source);
    const fields = this.rootEl.querySelectorAll('[data-config-field="true"]');
    fields.forEach((el) => {
      const tokens = JSON.parse(el.dataset.path);
      const original = getByPath(this.source, tokens);
      const parsed = parseValue(el.value, original);
      setByPath(updated, tokens, parsed);
    });

    const covariateFields = this.rootEl.querySelectorAll('[data-covariates-field="true"]');
    covariateFields.forEach((el) => {
      const tokens = JSON.parse(el.dataset.path);
      const selected = Array.from(el.querySelectorAll('input[type="checkbox"]:checked')).map((item) => item.value);
      setByPath(updated, tokens, selected);
    });

    return stripCommentKeysDeep(updated);
  }
}
