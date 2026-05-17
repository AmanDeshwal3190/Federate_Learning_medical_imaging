/**
 * NeuroScan AI — Clinical Diagnostic Dashboard JavaScript
 * Handles: Tab navigation, file upload, image zoom/pan, analysis API,
 * database matching carousel, annotation modal, + existing WebSocket/Chart.js.
 */

// ═══════════════════════════════════════════════════
// GLOBAL STATE
// ═══════════════════════════════════════════════════
let socket;
let accuracyChart, lossChart, comparisonChart;

// Clinical state
let uploadedFile = null;
let selectedDisease = 'alzheimer';
let viewerZoom = 1;
let viewerPanX = 0, viewerPanY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;
let modalZoom = 1;
let modalPanX = 0, modalPanY = 0;
let isModalPanning = false;
let modalPanStartX = 0, modalPanStartY = 0;

// Chart.js theme
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.borderColor = 'rgba(22, 33, 62, 0.5)';

const clientColors = {};
const colorPalette = [
    '#00d4ff', '#10b981', '#f59e0b', '#ef4444',
    '#7c3aed', '#ec4899', '#14b8a6', '#f97316'
];
function getRandomColor(index) {
    return colorPalette[index % colorPalette.length];
}

// ═══════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initTabNavigation();
    initUploadZone();
    initDiseaseToggle();
    initImageViewer();
    initAnalyzeButton();
    initModalControls();
    initCharts();
    initWebSocket();
    fetchInitialData();
});

// ═══════════════════════════════════════════════════
// 1. TAB NAVIGATION
// ═══════════════════════════════════════════════════
function initTabNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetView = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            document.querySelectorAll('.tab-content').forEach(v => v.classList.remove('active'));
            document.getElementById(`view-${targetView}`).classList.add('active');
        });
    });
}

// ═══════════════════════════════════════════════════
// 2. FILE UPLOAD (Drag & Drop + Click)
// ═══════════════════════════════════════════════════
function initUploadZone() {
    const zone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name-display');
    const fileRemove = document.getElementById('file-remove');
    const previewMini = document.getElementById('upload-preview-mini');

    // Click to open file dialog
    zone.addEventListener('click', () => fileInput.click());

    // Drag events
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelected(e.dataTransfer.files[0]);
        }
    });

    // File input change
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileSelected(fileInput.files[0]);
        }
    });

    // Remove file
    fileRemove.addEventListener('click', (e) => {
        e.stopPropagation();
        clearUploadedFile();
    });
}

function handleFileSelected(file) {
    uploadedFile = file;
    const fileName = document.getElementById('file-name-display');
    const fileInfo = document.getElementById('file-info');
    const previewMini = document.getElementById('upload-preview-mini');
    const btnAnalyze = document.getElementById('btn-analyze');
    const zone = document.getElementById('upload-zone');

    fileName.textContent = file.name;
    fileInfo.classList.add('visible');

    // Show mini preview
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewMini.src = e.target.result;
            previewMini.style.display = 'block';

            // Also show in main viewer
            showInViewer(e.target.result, file);
        };
        reader.readAsDataURL(file);
    } else {
        previewMini.style.display = 'none';
        // For non-image files, show a placeholder
        showInViewerPlaceholder(file);
    }

    // Enable analyze button
    const hospital = document.getElementById('hospital-name').value.trim();
    btnAnalyze.disabled = !hospital;

    // Add listener to hospital name too
    document.getElementById('hospital-name').addEventListener('input', () => {
        btnAnalyze.disabled = !(document.getElementById('hospital-name').value.trim() && uploadedFile);
    });

    showToast(`File loaded: ${file.name}`, 'success');
}

function clearUploadedFile() {
    uploadedFile = null;
    document.getElementById('file-info').classList.remove('visible');
    document.getElementById('upload-preview-mini').style.display = 'none';
    document.getElementById('file-input').value = '';
    document.getElementById('btn-analyze').disabled = true;

    // Reset viewer
    const viewerImg = document.getElementById('viewer-image');
    viewerImg.classList.remove('visible');
    document.getElementById('viewer-placeholder').style.display = 'flex';
    document.getElementById('zoom-level').style.display = 'none';
    document.getElementById('image-meta').style.display = 'none';
    resetViewerZoom();
}

function showInViewer(dataUrl, file) {
    const viewerImg = document.getElementById('viewer-image');
    const placeholder = document.getElementById('viewer-placeholder');
    const zoomBadge = document.getElementById('zoom-level');
    const meta = document.getElementById('image-meta');

    viewerImg.src = dataUrl;
    viewerImg.classList.add('visible');
    placeholder.style.display = 'none';
    zoomBadge.style.display = 'block';
    meta.style.display = 'flex';

    viewerImg.onload = () => {
        document.getElementById('meta-dimensions').textContent = `${viewerImg.naturalWidth} × ${viewerImg.naturalHeight}`;
        document.getElementById('meta-format').textContent = file.type || file.name.split('.').pop().toUpperCase();
        document.getElementById('meta-size').textContent = formatFileSize(file.size);
        const hospital = document.getElementById('hospital-name').value.trim();
        document.getElementById('meta-hospital').textContent = hospital || '—';
    };

    resetViewerZoom();
}

function showInViewerPlaceholder(file) {
    const viewerImg = document.getElementById('viewer-image');
    const placeholder = document.getElementById('viewer-placeholder');
    placeholder.innerHTML = `
        <span class="placeholder-icon">📄</span>
        <span>${file.name}</span>
        <span style="font-size: 0.8rem; color: var(--text-dim);">
            Medical file loaded — will be processed during analysis
        </span>
    `;
    viewerImg.classList.remove('visible');
}

// ═══════════════════════════════════════════════════
// 3. DISEASE TOGGLE
// ═══════════════════════════════════════════════════
function initDiseaseToggle() {
    const buttons = document.querySelectorAll('.disease-option');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            buttons.forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedDisease = btn.dataset.disease;
        });
    });
}

// ═══════════════════════════════════════════════════
// 4. IMAGE VIEWER (Zoom + Pan)
// ═══════════════════════════════════════════════════
function initImageViewer() {
    const viewer = document.getElementById('image-viewer');
    const img = document.getElementById('viewer-image');

    // Mouse wheel zoom
    viewer.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -0.1 : 0.1;
        viewerZoom = Math.max(0.2, Math.min(8, viewerZoom + delta));
        applyViewerTransform();
    });

    // Pan with mouse drag
    viewer.addEventListener('mousedown', (e) => {
        if (!img.classList.contains('visible')) return;
        isPanning = true;
        panStartX = e.clientX - viewerPanX;
        panStartY = e.clientY - viewerPanY;
        viewer.style.cursor = 'grabbing';
    });
    window.addEventListener('mousemove', (e) => {
        if (!isPanning) return;
        viewerPanX = e.clientX - panStartX;
        viewerPanY = e.clientY - panStartY;
        applyViewerTransform();
    });
    window.addEventListener('mouseup', () => {
        isPanning = false;
        document.getElementById('image-viewer').style.cursor = 'grab';
    });

    // Zoom buttons
    document.getElementById('zoom-in').addEventListener('click', () => {
        viewerZoom = Math.min(8, viewerZoom + 0.25);
        applyViewerTransform();
    });
    document.getElementById('zoom-out').addEventListener('click', () => {
        viewerZoom = Math.max(0.2, viewerZoom - 0.25);
        applyViewerTransform();
    });
    document.getElementById('zoom-reset').addEventListener('click', resetViewerZoom);
    document.getElementById('zoom-fit').addEventListener('click', fitToView);
}

function applyViewerTransform() {
    const img = document.getElementById('viewer-image');
    img.style.transform = `translate(${viewerPanX}px, ${viewerPanY}px) scale(${viewerZoom})`;
    document.getElementById('zoom-level').textContent = Math.round(viewerZoom * 100) + '%';
}

function resetViewerZoom() {
    viewerZoom = 1;
    viewerPanX = 0;
    viewerPanY = 0;
    applyViewerTransform();
}

function fitToView() {
    const viewer = document.getElementById('image-viewer');
    const img = document.getElementById('viewer-image');
    if (!img.naturalWidth) return;

    const wRatio = viewer.clientWidth / img.naturalWidth;
    const hRatio = viewer.clientHeight / img.naturalHeight;
    viewerZoom = Math.min(wRatio, hRatio) * 0.95;
    viewerPanX = 0;
    viewerPanY = 0;
    applyViewerTransform();
}

// ═══════════════════════════════════════════════════
// 5. ANALYZE BUTTON + API CALLS
// ═══════════════════════════════════════════════════
function initAnalyzeButton() {
    const btn = document.getElementById('btn-analyze');
    // Enable button when hospital name changes and file is present
    document.getElementById('hospital-name').addEventListener('input', () => {
        btn.disabled = !(document.getElementById('hospital-name').value.trim() && uploadedFile);
    });

    btn.addEventListener('click', runAnalysis);
}

async function runAnalysis() {
    if (!uploadedFile) return;
    const btn = document.getElementById('btn-analyze');
    const hospitalName = document.getElementById('hospital-name').value.trim();

    if (!hospitalName) {
        showToast('Please enter a hospital/client name', 'warning');
        return;
    }

    btn.classList.add('loading');
    btn.disabled = true;
    document.getElementById('viewer-processing').classList.add('visible');

    try {
        // Step 1: Upload
        const formData = new FormData();
        formData.append('file', uploadedFile);
        formData.append('hospital_name', hospitalName);
        formData.append('disease', selectedDisease);

        showToast('Uploading scan...', 'info');
        const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();

        if (!uploadRes.ok) {
            throw new Error(uploadData.error || 'Upload failed');
        }

        showToast('Running AI analysis...', 'info');

        // Step 2: Analyze
        const analyzeRes = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: uploadData.image_path,
                disease: selectedDisease,
                hospital_name: hospitalName,
                original_filename: uploadData.filename
            })
        });
        const analyzeData = await analyzeRes.json();

        // Step 3: Get database matches
        showToast('Matching against database...', 'info');
        const matchRes = await fetch('/api/match-database', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: uploadData.image_path,
                disease: selectedDisease
            })
        });
        const matchData = await matchRes.json();

        // Render results
        renderDatabaseMatches(matchData.matches || []);
        showAnalysisModal(analyzeData, uploadData.image_url);

        showToast('Analysis complete!', 'success');

    } catch (err) {
        console.error('Analysis failed:', err);
        showToast(`Error: ${err.message}`, 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
        document.getElementById('viewer-processing').classList.remove('visible');
    }
}

// ═══════════════════════════════════════════════════
// 6. DATABASE MATCH CAROUSEL
// ═══════════════════════════════════════════════════
function renderDatabaseMatches(matches) {
    const track = document.getElementById('carousel-track');
    const empty = document.getElementById('match-empty');
    const prevBtn = document.getElementById('carousel-prev');
    const nextBtn = document.getElementById('carousel-next');
    const dotsContainer = document.getElementById('carousel-dots');

    if (!matches || matches.length === 0) {
        track.style.display = 'none';
        prevBtn.style.display = 'none';
        nextBtn.style.display = 'none';
        empty.style.display = 'flex';
        dotsContainer.innerHTML = '';
        return;
    }

    empty.style.display = 'none';
    track.style.display = 'flex';
    prevBtn.style.display = 'flex';
    nextBtn.style.display = 'flex';
    track.innerHTML = '';
    dotsContainer.innerHTML = '';

    matches.forEach((match, idx) => {
        const card = document.createElement('div');
        card.className = 'match-card';
        card.innerHTML = `
            <img class="match-card-image" src="${match.image_url}" alt="${match.label}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 160%22><rect fill=%22%230a0a15%22 width=%22200%22 height=%22160%22/><text fill=%22%2364748b%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22 font-size=%2214%22>No Preview</text></svg>'">
            <div class="match-card-badge">${match.similarity}%</div>
            <div class="match-card-info">
                <div class="match-card-label">${match.label}</div>
                <div class="match-card-score">Similarity: ${match.similarity}%</div>
            </div>
        `;
        card.addEventListener('click', () => {
            // Show enlarged in viewer
            const viewerImg = document.getElementById('viewer-image');
            viewerImg.src = match.image_url;
            viewerImg.classList.add('visible');
            document.getElementById('viewer-placeholder').style.display = 'none';
            resetViewerZoom();
            showToast(`Viewing: ${match.label}`, 'info');
        });
        track.appendChild(card);

        // Dot
        const dot = document.createElement('div');
        dot.className = `carousel-dot${idx === 0 ? ' active' : ''}`;
        dotsContainer.appendChild(dot);
    });

    // Carousel navigation
    let carouselPos = 0;
    const slideWidth = 220;

    prevBtn.onclick = () => {
        carouselPos = Math.max(0, carouselPos - 1);
        track.style.transform = `translateX(-${carouselPos * slideWidth}px)`;
        updateDots(carouselPos);
    };
    nextBtn.onclick = () => {
        const maxPos = Math.max(0, matches.length - Math.floor(track.parentElement.clientWidth / slideWidth));
        carouselPos = Math.min(maxPos, carouselPos + 1);
        track.style.transform = `translateX(-${carouselPos * slideWidth}px)`;
        updateDots(carouselPos);
    };

    function updateDots(pos) {
        dotsContainer.querySelectorAll('.carousel-dot').forEach((d, i) => {
            d.classList.toggle('active', i === pos);
        });
    }
}

// ═══════════════════════════════════════════════════
// 7. ANALYSIS MODAL WITH ANNOTATIONS
// ═══════════════════════════════════════════════════
function initModalControls() {
    document.getElementById('modal-close').addEventListener('click', closeModal);
    document.getElementById('analysis-modal-backdrop').addEventListener('click', (e) => {
        if (e.target === document.getElementById('analysis-modal-backdrop')) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });

    // Modal zoom buttons
    document.getElementById('modal-zoom-in').addEventListener('click', () => {
        modalZoom = Math.min(5, modalZoom + 0.25);
        applyModalZoom();
    });
    document.getElementById('modal-zoom-out').addEventListener('click', () => {
        modalZoom = Math.max(0.3, modalZoom - 0.25);
        applyModalZoom();
    });

    // Modal scroll-wheel zoom
    const modalArea = document.getElementById('modal-image-area');
    modalArea.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -0.15 : 0.15;
        modalZoom = Math.max(0.3, Math.min(5, modalZoom + delta));
        applyModalZoom();
    });

    // Modal pan with mouse drag
    modalArea.addEventListener('mousedown', (e) => {
        isModalPanning = true;
        modalPanStartX = e.clientX - modalPanX;
        modalPanStartY = e.clientY - modalPanY;
        modalArea.style.cursor = 'grabbing';
        e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
        if (!isModalPanning) return;
        modalPanX = e.clientX - modalPanStartX;
        modalPanY = e.clientY - modalPanStartY;
        applyModalZoom();
    });
    window.addEventListener('mouseup', () => {
        if (isModalPanning) {
            isModalPanning = false;
            const modalArea = document.getElementById('modal-image-area');
            if (modalArea) modalArea.style.cursor = 'crosshair';
        }
    });
}

function applyModalZoom() {
    const container = document.getElementById('modal-image-container');
    container.style.transform = `translate(${modalPanX}px, ${modalPanY}px) scale(${modalZoom})`;
}

function showAnalysisModal(data, imageUrl) {
    const backdrop = document.getElementById('analysis-modal-backdrop');
    const modalImg = document.getElementById('modal-image');
    const container = document.getElementById('modal-image-container');
    const regionsList = document.getElementById('regions-list');
    const subtitle = document.getElementById('diagnosis-subtitle');

    // Set the image
    if (data.highlighted_image_url) {
        modalImg.src = data.highlighted_image_url;
    } else {
        modalImg.src = imageUrl;
    }

    // Reset zoom + pan state for modal
    modalZoom = 1;
    modalPanX = 0;
    modalPanY = 0;
    container.style.transform = 'translate(0px, 0px) scale(1)';

    // Clear old markers
    container.querySelectorAll('.annotation-marker').forEach(m => m.remove());

    // Set subtitle and header context based on result
    const diseaseLabel = selectedDisease === 'alzheimer' ? "Alzheimer's Disease" : "Brain Tumor";
    const regionsHeader = document.querySelector('.regions-header h3');
    
    if (data.disease_detected) {
        subtitle.textContent = `${diseaseLabel} — ${data.diagnosis || 'Analysis Complete'}`;
        subtitle.style.color = '';
        if (regionsHeader) regionsHeader.innerHTML = '🧬 Affected Brain Regions';
    } else {
        subtitle.textContent = `${diseaseLabel} Screening — ${data.diagnosis || 'HEALTHY'}`;
        subtitle.style.color = '#10b981';
        if (regionsHeader) regionsHeader.innerHTML = '🧬 Brain Region Health Report';
    }

    // Summary
    document.getElementById('summary-diagnosis').textContent = data.diagnosis || '—';
    document.getElementById('summary-diagnosis').className = `summary-value ${data.disease_detected ? 'positive' : 'negative'}`;
    document.getElementById('summary-confidence').textContent = data.confidence ? `${data.confidence}%` : '—';
    
    const riskPct = data.disease_risk_percent !== undefined ? data.disease_risk_percent : null;
    if (data.disease_detected) {
        document.getElementById('summary-detection').textContent = '⚠️ Positive';
    } else {
        document.getElementById('summary-detection').textContent = `✅ Negative${riskPct !== null ? ' (' + riskPct + '% risk)' : ''}`;
    }
    document.getElementById('summary-detection').className = `summary-value ${data.disease_detected ? 'positive' : 'negative'}`;

    // Render region cards + markers
    regionsList.innerHTML = '';
    const regions = data.affected_regions || [];

    // If no disease detected, show healthy brain report with region breakdown
    if (!data.disease_detected) {
        const riskDisplay = riskPct !== null ? riskPct : '< 5';
        let healthyHTML = `
            <div style="display: flex; flex-direction: column; align-items: center; padding: 1.2rem 1rem 0.6rem; text-align: center; gap: 0.5rem; border-bottom: 1px solid rgba(16,185,129,0.2); margin-bottom: 0.75rem;">
                <div style="font-size: 2.5rem;">✅</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #10b981;">HEALTHY BRAIN — No Disease Detected</div>
                <div style="font-size: 0.9rem; color: #10b981; font-weight: 600;">
                    Overall Disease Probability: ${riskDisplay}% (Very Low)
                </div>
                <div style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.5; max-width: 360px;">
                    AI analysis confirms all brain regions are within normal parameters. 
                    No pathological indicators detected. Diagnosis: <strong style="color:#10b981;">NEGATIVE</strong>
                </div>
            </div>
        `;
        regionsList.innerHTML = healthyHTML;
    }

    // Helper to create all annotation markers on the container
    function createAnnotationMarkers() {
        // Remove any existing markers first
        container.querySelectorAll('.annotation-marker').forEach(m => m.remove());

        regions.forEach((r, i) => {
            const marker = document.createElement('div');
            marker.className = `annotation-marker${!data.disease_detected ? ' healthy-marker' : ''}`;
            marker.dataset.index = i + 1;
            marker.style.left = `${r.x_percent || (20 + i * 15)}%`;
            marker.style.top = `${r.y_percent || (30 + i * 10)}%`;
            marker.title = r.name;
            if (!data.disease_detected) {
                marker.style.background = 'rgba(16, 185, 129, 0.9)';
                marker.style.boxShadow = '0 0 12px rgba(16, 185, 129, 0.5)';
            }
            container.appendChild(marker);
        });
    }

    regions.forEach((region, idx) => {
        // Card
        const card = document.createElement('div');
        card.className = 'region-card';
        card.dataset.index = idx;

        const isHealthy = region.severity === 'Normal';
        const severityClass = isHealthy ? 'severity-normal' :
            region.severity === 'High' ? 'severity-high' :
            region.severity === 'Moderate' ? 'severity-moderate' : 'severity-low';

        const riskBadge = region.risk_percent !== undefined 
            ? `<span style="font-size: 0.7rem; color: #10b981; font-weight: 600; margin-left: 4px;">(${region.risk_percent}% risk)</span>` 
            : '';

        const conditionIcon = isHealthy ? '✅' : '🔸';
        const severityLabel = isHealthy ? 'Normal' : region.severity;

        card.innerHTML = `
            <div class="region-header">
                <div class="region-name">
                    <span class="region-index" ${isHealthy ? 'style="background: #10b981;"' : ''}>${idx + 1}</span>
                    ${region.name} ${riskBadge}
                </div>
                <span class="region-severity ${severityClass}">${severityLabel}</span>
            </div>
            <div class="region-description">${region.description}</div>
            <div class="region-condition">
                ${conditionIcon} ${region.condition}
            </div>
        `;

        card.addEventListener('mouseenter', () => {
            card.classList.add('highlighted');
            const marker = container.querySelector(`.annotation-marker[data-index="${idx + 1}"]`);
            if (marker) marker.style.transform = 'translate(-50%, -50%) scale(1.5)';
        });
        card.addEventListener('mouseleave', () => {
            card.classList.remove('highlighted');
            const marker = container.querySelector(`.annotation-marker[data-index="${idx + 1}"]`);
            if (marker) marker.style.transform = 'translate(-50%, -50%) scale(1)';
        });

        regionsList.appendChild(card);
    });

    // Place markers once image is loaded (so container has the right size)
    modalImg.onload = () => {
        createAnnotationMarkers();
    };
    // If image is already cached/loaded
    if (modalImg.complete && modalImg.naturalWidth > 0) {
        createAnnotationMarkers();
    }

    backdrop.classList.add('visible');
}

function closeModal() {
    document.getElementById('analysis-modal-backdrop').classList.remove('visible');
}

// ═══════════════════════════════════════════════════
// 8. WEBSOCKET (existing, preserved)
// ═══════════════════════════════════════════════════
function initWebSocket() {
    socket = io();

    const connectionStatus = document.getElementById('connection-status');
    const headerStatusText = document.getElementById('header-status-text');

    socket.on('connect', () => {
        connectionStatus.className = 'status-dot tooltip connected';
        headerStatusText.textContent = 'Connected (Live)';
        showToast('Connected to FL Server', 'success');
    });

    socket.on('disconnect', () => {
        connectionStatus.className = 'status-dot tooltip disconnected';
        headerStatusText.textContent = 'Disconnected';
        showToast('Disconnected from server. Retrying...', 'error');
    });

    socket.on('reconnect', () => {
        showToast('Reconnected to FL Server', 'success');
    });

    socket.on('initial_data', (data) => {
        if (data.rounds && data.rounds.length > 0) {
            document.getElementById('acc-loading').classList.add('hidden');
            document.getElementById('loss-loading').classList.add('hidden');
            resetCharts();
            data.rounds.forEach(round => updateCharts(round));
        }
        if (data.clients) {
            updateClientTable(data.clients);
        }
    });

    socket.on('new_metrics', (data) => {
        document.getElementById('acc-loading').classList.add('hidden');
        document.getElementById('loss-loading').classList.add('hidden');
        updateCharts(data);
        if (data.clients) {
            updateClientTable(data.clients);
        }
    });

    socket.on('training_status', (data) => {
        updateTrainingStatus(data);
    });
}

// ═══════════════════════════════════════════════════
// 9. CHART.JS (existing, preserved)
// ═══════════════════════════════════════════════════
function initCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 500 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                position: 'bottom',
                labels: { usePointStyle: true, boxWidth: 8 }
            },
            tooltip: {
                backgroundColor: 'rgba(15, 15, 35, 0.9)',
                titleColor: '#fff',
                bodyColor: '#e2e8f0',
                borderColor: 'rgba(0, 212, 255, 0.3)',
                borderWidth: 1
            }
        },
        scales: {
            x: { title: { display: true, text: 'FL Round' }, grid: { display: false } },
            y: { grid: { color: 'rgba(22, 33, 62, 0.5)' } }
        }
    };

    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    accuracyChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Global Accuracy', data: [],
                borderColor: '#00d4ff', backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 3, tension: 0.3, fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'Accuracy' }, min: 0 } }
        }
    });

    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Global Loss', data: [],
                borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)',
                borderWidth: 3, tension: 0.3, fill: true
            }]
        },
        options: {
            ...commonOptions,
            scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'Loss' }, min: 0 } }
        }
    });

    const ctxComp = document.getElementById('comparisonChart').getContext('2d');
    comparisonChart = new Chart(ctxComp, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Loss'],
            datasets: [
                { label: 'Federated', data: [0, 0], backgroundColor: 'rgba(16, 185, 129, 0.8)', borderRadius: 4 },
                { label: 'Centralized Baseline', data: [0, 0], backgroundColor: 'rgba(148, 163, 184, 0.5)', borderRadius: 4 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'bottom' } },
            scales: { y: { beginAtZero: true } }
        }
    });
}

function resetCharts() {
    accuracyChart.data.labels = [];
    accuracyChart.data.datasets = [accuracyChart.data.datasets[0]];
    accuracyChart.data.datasets[0].data = [];
    lossChart.data.labels = [];
    lossChart.data.datasets = [lossChart.data.datasets[0]];
    lossChart.data.datasets[0].data = [];
}

function updateCharts(roundData) {
    const round = roundData.round;
    if (!accuracyChart.data.labels.includes(round)) {
        accuracyChart.data.labels.push(round);
        lossChart.data.labels.push(round);
    }

    const idx = accuracyChart.data.labels.indexOf(round);
    accuracyChart.data.datasets[0].data[idx] = roundData.global_accuracy;
    lossChart.data.datasets[0].data[idx] = roundData.global_loss;

    if (roundData.clients) {
        for (const [clientId, clientMetrics] of Object.entries(roundData.clients)) {
            if (!clientColors[clientId]) {
                clientColors[clientId] = getRandomColor(Object.keys(clientColors).length);
            }

            let accDataset = accuracyChart.data.datasets.find(ds => ds.label === clientId);
            if (!accDataset) {
                accDataset = {
                    label: clientId, data: new Array(accuracyChart.data.labels.length).fill(null),
                    borderColor: clientColors[clientId], borderWidth: 1.5, borderDash: [5, 5], tension: 0.3, pointRadius: 2
                };
                accuracyChart.data.datasets.push(accDataset);
            }
            accDataset.data[idx] = clientMetrics.accuracy;

            let lossDataset = lossChart.data.datasets.find(ds => ds.label === clientId);
            if (!lossDataset) {
                lossDataset = {
                    label: clientId, data: new Array(lossChart.data.labels.length).fill(null),
                    borderColor: clientColors[clientId], borderWidth: 1.5, borderDash: [5, 5], tension: 0.3, pointRadius: 2
                };
                lossChart.data.datasets.push(lossDataset);
            }
            lossDataset.data[idx] = clientMetrics.loss;
        }
    }

    accuracyChart.update();
    lossChart.update();

    animateValue('stat-global-acc', 0, roundData.global_accuracy * 100, 500, true);

    const allAcc = accuracyChart.data.datasets[0].data.filter(v => v !== null);
    if (allAcc.length > 0) {
        const best = Math.max(...allAcc);
        const bestIdx = accuracyChart.data.datasets[0].data.indexOf(best);
        const bestRound = accuracyChart.data.labels[bestIdx];
        document.getElementById('stat-best-acc').innerText = (best * 100).toFixed(1) + '%';
        document.getElementById('stat-best-acc-round').innerText = bestRound;
    }

    fetchComparisonData();
}

function updateClientTable(clientsData) {
    const tbody = document.getElementById('client-table-body');
    const clientIds = Object.keys(clientsData);
    if (clientIds.length === 0) return;

    tbody.innerHTML = '';
    clientIds.forEach(id => {
        const c = clientsData[id];
        const tr = document.createElement('tr');
        const statusClass = c.status === 'active' ? 'connected' : (c.status === 'training' ? 'training' : 'disconnected');
        tr.innerHTML = `
            <td><div class="status-cell"><span class="status-dot ${statusClass}"></span><span>${c.status || 'active'}</span></div></td>
            <td><strong>${id}</strong></td>
            <td>${formatNumber(c.samples || c.num_samples || 0)}</td>
            <td>${(c.accuracy !== undefined ? (c.accuracy * 100).toFixed(2) + '%' : '-')}</td>
            <td>${(c.loss !== undefined ? c.loss.toFixed(4) : '-')}</td>
            <td style="font-size: 0.8rem; color: var(--text-muted);">${formatTimeRecord(c.last_update)}</td>
        `;
        tbody.appendChild(tr);
    });
    document.getElementById('stat-active-clients').innerText = clientIds.length;
}

function updateTrainingStatus(data) {
    if (data.current_round) document.getElementById('stat-current-round').innerText = data.current_round;
    if (data.total_rounds) document.getElementById('stat-total-rounds').innerText = data.total_rounds;
}

// ═══════════════════════════════════════════════════
// 10. DATA FETCHING (existing, preserved)
// ═══════════════════════════════════════════════════
async function fetchInitialData() {
    try {
        const [statusRes, metricsRes, clientsRes, modelRes] = await Promise.all([
            fetch('/api/status'),
            fetch('/api/metrics'),
            fetch('/api/clients'),
            fetch('/api/model-info')
        ]);

        if (statusRes.ok) {
            const status = await statusRes.json();
            updateTrainingStatus(status);
        }

        if (clientsRes.ok) {
            const clientsData = await clientsRes.json();
            const clientsMap = {};
            clientsData.clients.forEach(c => clientsMap[c.id] = c);
            updateClientTable(clientsMap);
        }

        if (modelRes.ok) {
            const modelInfo = await modelRes.json();
            document.getElementById('model-info-text').innerText = `${modelInfo.model_type} - ${modelInfo.architecture} (${formatNumber(modelInfo.parameters)} params)`;
        }

        fetchComparisonData();
    } catch (e) {
        console.error("Failed to fetch initial data", e);
    }
}

async function fetchComparisonData() {
    try {
        const res = await fetch('/api/comparison');
        if (res.ok) {
            const data = await res.json();
            comparisonChart.data.datasets[0].data = [data.federated.accuracy, data.federated.loss];
            comparisonChart.data.datasets[1].data = [data.centralized.accuracy, data.centralized.loss];
            comparisonChart.update();
        }
    } catch (e) {}
}

// ═══════════════════════════════════════════════════
// 11. UTILITIES
// ═══════════════════════════════════════════════════
function formatNumber(num) {
    if (num === undefined || num === null) return "0";
    return new Intl.NumberFormat().format(num);
}

function formatTimeRecord(isoString) {
    if (!isoString) return "Never";
    try {
        const d = new Date(isoString);
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch (e) { return isoString; }
}

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (bytes >= 1024 && i < units.length - 1) {
        bytes /= 1024;
        i++;
    }
    return bytes.toFixed(1) + ' ' + units[i];
}

function animateValue(id, start, end, duration, isPercentage = false) {
    const obj = document.getElementById(id);
    if (!obj) return;
    if (typeof end === 'string') end = parseFloat(end);

    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = start + progress * (end - start);
        obj.innerHTML = isPercentage ? current.toFixed(1) + '%' : Math.floor(current);
        if (progress < 1) window.requestAnimationFrame(step);
    };
    window.requestAnimationFrame(step);
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';
    if (type === 'warning') icon = '⚠️';

    toast.innerHTML = `
        <div style="font-size: 1.5rem;">${icon}</div>
        <div class="toast-content">
            <span class="toast-title">${type.charAt(0).toUpperCase() + type.slice(1)}</span>
            <span class="toast-message">${message}</span>
        </div>
    `;

    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => {
            if (container.contains(toast)) container.removeChild(toast);
        }, 300);
    }, 4000);
}
