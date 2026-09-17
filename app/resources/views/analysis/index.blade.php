<!DOCTYPE html>
<html lang="en" class="h-full">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>Hybrid IF-SVM — Intrusion Analysis</title>

    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        [x-cloak] { display: none; }
        .drop-active { border-color: #6366f1; background-color: #eef2ff; }
    </style>
</head>
<body class="h-full bg-slate-100 text-slate-800 antialiased">

<div class="mx-auto max-w-6xl px-4 py-8">

    {{-- ---------------------------------------------------------------- --}}
    {{-- Header                                                           --}}
    {{-- ---------------------------------------------------------------- --}}
    <header class="mb-6">
        <h1 class="text-2xl font-semibold tracking-tight">
            Hybrid Isolation Forest–SVM
        </h1>
        <p class="mt-1 text-sm text-slate-500">
            Upload network flow records and compare the Shukla baseline
            against the proposed hybrid model.
        </p>
    </header>

    @unless($bundleReady)
        <div class="mb-6 rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
            <p class="font-medium">No model bundle loaded.</p>
            <p class="mt-1">
                Run <code class="rounded bg-amber-100 px-1">ml/export_inference_bundle.py</code>
                at the end of a training run, then place
                <code class="rounded bg-amber-100 px-1">inference_bundle.joblib</code>
                in <code class="rounded bg-amber-100 px-1">storage/app/ml/</code>.
                Uploads will fail until then.
            </p>
        </div>
    @endunless

    {{-- ---------------------------------------------------------------- --}}
    {{-- Upload                                                           --}}
    {{-- ---------------------------------------------------------------- --}}
    <section
        id="dropzone"
        class="rounded-xl border-2 border-dashed border-slate-300 bg-white p-8 transition-colors"
    >
        <div class="text-center">
            <svg class="mx-auto h-10 w-10 text-slate-400" fill="none"
                 viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5">
                <path stroke-linecap="round" stroke-linejoin="round"
                      d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 7.5 7.5 12M12 7.5V21"/>
            </svg>

            <p class="mt-3 text-sm font-medium text-slate-700">
                Drag CSV files or a whole folder here
            </p>
            <p class="mt-1 text-xs text-slate-500">
                Up to {{ $maxFiles }} files, {{ $maxFileMb }} MB each
            </p>

            <div class="mt-4 flex items-center justify-center gap-3">
                <button type="button" id="pickFiles"
                        class="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500">
                    Choose files
                </button>
                <button type="button" id="pickFolder"
                        class="rounded-md border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50">
                    Choose folder
                </button>
            </div>

            <input id="fileInput" type="file" accept=".csv,text/csv" multiple class="hidden">
            <input id="folderInput" type="file" webkitdirectory directory multiple class="hidden">
        </div>

        {{-- queued files --}}
        <div id="fileList" class="mt-6 hidden divide-y divide-slate-100 rounded-lg border border-slate-200"></div>

        <div id="actions" class="mt-4 hidden items-center justify-between">
            <p id="queueSummary" class="text-sm text-slate-500"></p>
            <div class="flex gap-2">
                <button type="button" id="clearBtn"
                        class="rounded-md border border-slate-300 px-3 py-2 text-sm text-slate-600 hover:bg-slate-50">
                    Clear
                </button>
                <button type="button" id="analyseBtn"
                        class="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-40">
                    Analyse
                </button>
            </div>
        </div>

        <div id="progress" class="mt-4 hidden">
            <div class="h-1.5 w-full overflow-hidden rounded-full bg-slate-200">
                <div id="progressBar" class="h-full w-1/3 animate-pulse rounded-full bg-indigo-500"></div>
            </div>
            <p id="progressText" class="mt-2 text-center text-xs text-slate-500"></p>
        </div>

        <div id="uploadError" class="mt-4 hidden rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800"></div>
    </section>

    {{-- ---------------------------------------------------------------- --}}
    {{-- Results                                                          --}}
    {{-- ---------------------------------------------------------------- --}}
    <section id="results" class="mt-8 hidden">

        <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
            <div>
                <h2 class="text-lg font-semibold">Results</h2>
                <p id="resultScope" class="text-sm text-slate-500"></p>
            </div>
            <select id="fileSelect"
                    class="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm">
            </select>
        </div>

        <div class="grid gap-5 md:grid-cols-3">

            <article class="rounded-xl border border-slate-200 bg-white p-5">
                <div class="flex items-baseline justify-between">
                    <h3 class="text-sm font-semibold uppercase tracking-wide text-slate-500">
                        Baseline
                    </h3>
                    <span class="text-xs text-slate-400">Isolation Forest</span>
                </div>
                <div id="panelBaseline" class="mt-4"></div>
            </article>

            <article class="rounded-xl border-2 border-indigo-200 bg-white p-5">
                <div class="flex items-baseline justify-between">
                    <h3 class="text-sm font-semibold uppercase tracking-wide text-indigo-600">
                        Hybrid
                    </h3>
                    <span class="text-xs text-slate-400">IF + SVM</span>
                </div>
                <div id="panelHybrid" class="mt-4"></div>
            </article>
        </div>

            <article class="rounded-xl border border-slate-200 bg-white p-5">
            <div class="flex items-baseline justify-between">
                <h3 class="text-sm font-semibold uppercase tracking-wide text-slate-500">
                    Standalone SVM
                </h3>
                <span class="text-xs text-slate-400">
                    reference — isolates the fusion effect
                </span>
            </div>
            <div id="panelSvm" class="mt-4"></div>
            </article>
        </div>

        {{-- per-row preview --}}
        <div class="mt-6 rounded-xl border border-slate-200 bg-white">
            <div class="border-b border-slate-100 px-5 py-3">
                <h3 class="text-sm font-semibold text-slate-700">
                    Row preview
                    <span class="font-normal text-slate-400">(first 50)</span>
                </h3>
            </div>
            <div class="overflow-x-auto">
                <table class="min-w-full text-sm">
                    <thead class="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500">
                        <tr id="previewHead"></tr>
                    </thead>
                    <tbody id="previewBody" class="divide-y divide-slate-100"></tbody>
                </table>
            </div>
        </div>

        <p id="modelMeta" class="mt-4 text-xs text-slate-400"></p>
    </section>
</div>

<script>
(function () {
    'use strict';

    const MAX_FILES = {{ $maxFiles }};
    const csrf = document.querySelector('meta[name="csrf-token"]').content;

    let queue = [];        // {file, path}
    let payload = null;    // last server response

    const el = (id) => document.getElementById(id);

    const dropzone   = el('dropzone');
    const fileList   = el('fileList');
    const actions    = el('actions');
    const progress   = el('progress');
    const results    = el('results');
    const fileSelect = el('fileSelect');

    // ------------------------------------------------------------- upload

    el('pickFiles').addEventListener('click', () => el('fileInput').click());
    el('pickFolder').addEventListener('click', () => el('folderInput').click());

    el('fileInput').addEventListener('change', (e) => addFiles(e.target.files));
    el('folderInput').addEventListener('change', (e) => addFiles(e.target.files));

    ['dragenter', 'dragover'].forEach((type) => {
        dropzone.addEventListener(type, (e) => {
            e.preventDefault();
            dropzone.classList.add('drop-active');
        });
    });

    ['dragleave', 'drop'].forEach((type) => {
        dropzone.addEventListener(type, (e) => {
            e.preventDefault();
            if (type === 'dragleave' && dropzone.contains(e.relatedTarget)) return;
            dropzone.classList.remove('drop-active');
        });
    });

    dropzone.addEventListener('drop', async (e) => {
        e.preventDefault();

        const items = e.dataTransfer.items;

        // A dropped folder only yields its contents through the entries API.
        if (items && items.length && items[0].webkitGetAsEntry) {
            const files = [];
            const roots = [];

            for (const item of items) {
                const entry = item.webkitGetAsEntry();
                if (entry) roots.push(entry);
            }

            for (const root of roots) {
                await walkEntry(root, '', files);
            }

            addFiles(files);
        } else {
            addFiles(e.dataTransfer.files);
        }
    });

    function walkEntry(entry, prefix, out) {
        return new Promise((resolve) => {
            if (entry.isFile) {
                entry.file((file) => {
                    file._path = prefix + file.name;
                    out.push(file);
                    resolve();
                }, resolve);
                return;
            }

            const reader = entry.createReader();
            const all = [];

            const readBatch = () => {
                reader.readEntries(async (entries) => {
                    if (!entries.length) {
                        for (const child of all) {
                            await walkEntry(child, prefix + entry.name + '/', out);
                        }
                        resolve();
                        return;
                    }
                    all.push(...entries);
                    readBatch();
                }, resolve);
            };

            readBatch();
        });
    }

    function addFiles(list) {
        const incoming = Array.from(list).filter(
            (f) => /\.csv$/i.test(f.name)
        );

        const skipped = Array.from(list).length - incoming.length;

        for (const file of incoming) {
            if (queue.length >= MAX_FILES) break;

            const path = file._path || file.webkitRelativePath || file.name;
            if (queue.some((q) => q.path === path)) continue;

            queue.push({ file, path });
        }

        if (skipped > 0) {
            showError(`${skipped} non-CSV file(s) were ignored.`);
        } else {
            hideError();
        }

        renderQueue();
    }

    function renderQueue() {
        if (!queue.length) {
            fileList.classList.add('hidden');
            actions.classList.add('hidden');
            return;
        }

        fileList.classList.remove('hidden');
        actions.classList.remove('hidden');
        actions.classList.add('flex');

        fileList.innerHTML = queue.map((item, i) => `
            <div class="flex items-center justify-between px-4 py-2.5">
                <div class="min-w-0">
                    <p class="truncate text-sm text-slate-700">${escapeHtml(item.path)}</p>
                    <p class="text-xs text-slate-400">${formatBytes(item.file.size)}</p>
                </div>
                <button type="button" data-remove="${i}"
                        class="ml-4 shrink-0 text-xs text-slate-400 hover:text-rose-600">
                    Remove
                </button>
            </div>
        `).join('');

        const bytes = queue.reduce((sum, q) => sum + q.file.size, 0);
        el('queueSummary').textContent =
            `${queue.length} file${queue.length === 1 ? '' : 's'} · ${formatBytes(bytes)}`;

        fileList.querySelectorAll('[data-remove]').forEach((btn) => {
            btn.addEventListener('click', () => {
                queue.splice(Number(btn.dataset.remove), 1);
                renderQueue();
            });
        });
    }

    el('clearBtn').addEventListener('click', () => {
        queue = [];
        results.classList.add('hidden');
        hideError();
        renderQueue();
    });

    // ------------------------------------------------------------ analyse

    el('analyseBtn').addEventListener('click', async () => {
        if (!queue.length) return;

        const body = new FormData();
        queue.forEach((item, i) => {
            body.append('files[]', item.file);
            body.append(`paths[${i}]`, item.path);
        });

        setBusy(true, `Analysing ${queue.length} file(s)…`);
        hideError();

        try {
            const response = await fetch('{{ route('analysis.analyse') }}', {
                method: 'POST',
                headers: { 'X-CSRF-TOKEN': csrf, 'Accept': 'application/json' },
                body,
            });

            const raw = await response.text();
            let data;

            try {
                data = JSON.parse(raw);
            } catch {
                const detail = raw.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
                throw new Error(
                    `The server returned HTML instead of JSON (${response.status}). ` +
                    (detail ? detail.slice(0, 240) : 'Check the PHP upload limits and Laravel log.')
                );
            }

            if (!response.ok) {
                const messages = data.errors
                    ? Object.values(data.errors).flat().join(' ')
                    : (data.message || 'The upload was rejected.');
                throw new Error(messages);
            }

            payload = data;
            renderResults();
        } catch (err) {
            showError(err.message || 'Something went wrong.');
        } finally {
            setBusy(false);
        }
    });

    function setBusy(busy, text) {
        el('analyseBtn').disabled = busy;
        progress.classList.toggle('hidden', !busy);
        if (text) el('progressText').textContent = text;
    }

    // ------------------------------------------------------------ results

    function renderResults() {
        const ok = payload.results.filter((r) => r.ok);
        const failed = payload.results.filter((r) => !r.ok);

        if (!ok.length) {
            showError(failed.map((f) => `${f.file}: ${f.error}`).join(' | '));
            results.classList.add('hidden');
            return;
        }

        if (failed.length) {
            showError(
                `${failed.length} file(s) could not be analysed: ` +
                failed.map((f) => `${f.file} — ${f.error}`).join(' | ')
            );
        }

        fileSelect.innerHTML =
            `<option value="__all__">All files (${ok.length})</option>` +
            ok.map((r, i) => `<option value="${i}">${escapeHtml(r.file)}</option>`).join('');

        fileSelect.onchange = () => paint(fileSelect.value);

        // The thesis' official evaluation is the 82,332-row held-out test
        // partition. Prefer it when it is among the uploaded files instead
        // of silently aggregating train + test into 257,673 rows.
        const officialIndex = ok.findIndex((r) =>
            r.labelled && (r.official_test_candidate || Number(r.rows) === 82332)
        );
        const initialScope = officialIndex >= 0 ? String(officialIndex) : '__all__';
        fileSelect.value = initialScope;
        paint(initialScope);

        results.classList.remove('hidden');
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function paint(which) {
        const ok = payload.results.filter((r) => r.ok);
        const combined = which === '__all__';
        const source = combined ? payload.summary : ok[Number(which)];
        const labelled = combined ? payload.summary.labelled : source.labelled;

        const isOfficialTest = !combined && labelled &&
            (source.official_test_candidate || Number(source.rows) === 82332);

        el('resultScope').textContent = isOfficialTest
            ? `Official UNSW-NB15 held-out test · ${formatNumber(source.rows)} records`
            : (combined
                ? `Uploaded dataset analysis · ${payload.summary.files} file(s) · ${formatNumber(payload.summary.rows)} rows`
                : `Uploaded dataset analysis · ${source.file} · ${formatNumber(source.rows)} rows`);

        paintPanel('panelBaseline', source.baseline, labelled, null);
        paintPanel('panelSvm', source.svm, labelled, source.baseline);
        paintPanel('panelHybrid', source.hybrid, labelled, source.svm);

        paintPreview(combined ? ok[0] : source, combined);

        const meta = (combined ? ok[0] : source).meta || {};
        let metaText = meta.search_scoring
            ? `Model selection: ${meta.search_scoring} · threshold criterion: ${meta.threshold_criterion || 'n/a'}`
            : '';

        if (isOfficialTest && meta.official_test_results) {
            const expected = meta.official_test_results;
            const keys = ['baseline', 'svm', 'hybrid'];
            const metricKeys = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'balanced_accuracy', 'mcc'];
            const matches = keys.every((model) =>
                expected[model] && metricKeys.every((metric) =>
                    Math.abs(Number(source[model][metric]) - Number(expected[model][metric])) <= 1e-6
                ) && ['tn','fp','fn','tp'].every((cell) =>
                    Number(source[model][cell]) === Number(expected[model][cell])
                )
            );
            metaText += `${metaText ? ' · ' : ''}Kaggle reproduction: ${matches ? 'MATCH' : 'MISMATCH'}`;
        } else if (isOfficialTest && meta.test_accuracy_hybrid != null) {
            // Backward compatibility with older bundles that exported only
            // accuracy/MCC. A freshly exported bundle performs the full check.
            const matches =
                Math.abs(Number(source.baseline.accuracy) - Number(meta.test_accuracy_if)) <= 1e-6 &&
                Math.abs(Number(source.svm.accuracy) - Number(meta.test_accuracy_svm)) <= 1e-6 &&
                Math.abs(Number(source.hybrid.accuracy) - Number(meta.test_accuracy_hybrid)) <= 1e-6 &&
                Math.abs(Number(source.baseline.mcc) - Number(meta.test_mcc_if)) <= 1e-6 &&
                Math.abs(Number(source.svm.mcc) - Number(meta.test_mcc_svm)) <= 1e-6 &&
                Math.abs(Number(source.hybrid.mcc) - Number(meta.test_mcc_hybrid)) <= 1e-6;
            metaText += `${metaText ? ' · ' : ''}Kaggle reproduction: ${matches ? 'MATCH' : 'MISMATCH'} (accuracy/MCC)`;
        }

        el('modelMeta').textContent = metaText;
    }

    function paintPanel(id, data, labelled, compareTo) {
        if (!data) { el(id).innerHTML = ''; return; }

        if (!labelled) {
            el(id).innerHTML = `
                <p class="text-3xl font-semibold">${formatPercent(data.attack_rate)}</p>
                <p class="mt-1 text-sm text-slate-500">flagged as attack</p>
                <dl class="mt-4 grid grid-cols-2 gap-3 text-sm">
                    ${stat('Attacks', formatNumber(data.attacks))}
                    ${stat('Normal', formatNumber(data.normal))}
                </dl>
                <p class="mt-4 text-xs text-slate-400">
                    No <code>label</code> column, so accuracy cannot be computed.
                    Add one to score the models.
                </p>
            `;
            return;
        }

        const delta = compareTo
            ? deltaBadge(data.mcc - compareTo.mcc)
            : '';

        el(id).innerHTML = `
            <div class="flex items-end gap-3">
                <p class="text-3xl font-semibold">${formatPercent(data.accuracy)}</p>
                <p class="pb-1 text-sm text-slate-500">accuracy</p>
            </div>
            <dl class="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                ${stat('MCC', data.mcc.toFixed(4), delta)}
                ${stat('F1', data.f1.toFixed(4))}
                ${stat('Recall', formatPercent(data.recall))}
                ${stat('Specificity', formatPercent(data.specificity))}
                ${stat('Precision', formatPercent(data.precision))}
                ${stat('Balanced acc.', formatPercent(data.balanced_accuracy))}
            </dl>
            <div class="mt-4 grid grid-cols-4 gap-2 text-center text-xs">
                ${cell('TN', data.tn)}${cell('FP', data.fp)}
                ${cell('FN', data.fn)}${cell('TP', data.tp)}
            </div>
        `;
    }

    function stat(label, value, extra = '') {
        return `
            <div>
                <dt class="text-xs uppercase tracking-wide text-slate-400">${label}</dt>
                <dd class="mt-0.5 font-medium text-slate-800">${value} ${extra}</dd>
            </div>`;
    }

    function cell(label, value) {
        return `
            <div class="rounded-md bg-slate-50 py-2">
                <p class="text-[10px] uppercase tracking-wide text-slate-400">${label}</p>
                <p class="font-medium text-slate-700">${formatNumber(value)}</p>
            </div>`;
    }

    function deltaBadge(diff) {
        if (Math.abs(diff) < 0.0005) {
            return `<span class="ml-1 text-xs text-slate-400">±0</span>`;
        }
        const up = diff > 0;
        const cls = up ? 'text-emerald-600' : 'text-rose-600';
        return `<span class="ml-1 text-xs ${cls}">${up ? '+' : ''}${diff.toFixed(4)}</span>`;
    }

    function paintPreview(result, combined) {
        const rows = result.preview || [];
        if (!rows.length) return;

        const hasLabel = 'label' in rows[0];

        const headers = ['Row', 'IF score', 'Baseline', 'SVM', 'Hybrid']
            .concat(hasLabel ? ['Actual'] : []);

        el('previewHead').innerHTML =
            headers.map((h) => `<th class="px-5 py-2 font-medium">${h}</th>`).join('');

        el('previewBody').innerHTML = rows.map((r) => `
            <tr class="${hasLabel && r.hybrid !== r.label ? 'bg-rose-50/60' : ''}">
                <td class="px-5 py-2 text-slate-400">${r.row}</td>
                <td class="px-5 py-2 tabular-nums">${r.if_score.toFixed(4)}</td>
                <td class="px-5 py-2">${tag(r.baseline)}</td>
                <td class="px-5 py-2">${tag(r.svm)}</td>
                <td class="px-5 py-2">${tag(r.hybrid)}</td>
                ${hasLabel ? `<td class="px-5 py-2">${tag(r.label)}</td>` : ''}
            </tr>
        `).join('');

        if (combined) {
            el('previewHead').insertAdjacentHTML('beforeend',
                `<th class="px-5 py-2 text-right font-normal text-slate-400">${escapeHtml(result.file)}</th>`);
            el('previewBody').querySelectorAll('tr').forEach((tr) => {
                tr.insertAdjacentHTML('beforeend', '<td></td>');
            });
        }
    }

    function tag(value) {
        return value === 1
            ? '<span class="rounded bg-rose-100 px-1.5 py-0.5 text-xs font-medium text-rose-700">attack</span>'
            : '<span class="rounded bg-emerald-100 px-1.5 py-0.5 text-xs font-medium text-emerald-700">normal</span>';
    }

    // ------------------------------------------------------------ helpers

    function showError(message) {
        const box = el('uploadError');
        box.textContent = message;
        box.classList.remove('hidden');
    }

    function hideError() {
        el('uploadError').classList.add('hidden');
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1048576).toFixed(1) + ' MB';
    }

    const formatNumber = (n) => Number(n || 0).toLocaleString();
    const formatPercent = (n) => (Number(n || 0) * 100).toFixed(2) + '%';

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
})();
</script>
</body>
</html>
