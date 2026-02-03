/**
 * Human Tracking System - Frontend Controller
 * Handles video streaming, stats polling, and UI interactions.
 */

const API_URL = 'http://127.0.0.1:8000';

// State
let statsInterval = null;
let idleTimer = null;
let connectionCheckInterval = null;
let isConnected = false;
let currentVideo = '1.mp4';
let videoInfo = { process_width: 640, process_height: 360 };

// Configuration
const IDLE_TIMEOUT = 5000;
const STATS_POLL_INTERVAL = 500;  // Faster polling for real-time feel
const CONNECTION_CHECK_INTERVAL = 3000;
const RECONNECT_DELAY = 2000;

// =============================================================================
// Initialization
// =============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('[FYP] Initializing Human Tracking System...');

    // Check backend connection first
    await checkConnection();

    // Start stats polling
    startStatsPolling();

    // Start connection monitoring
    startConnectionMonitor();

    // Setup video container interactions
    setupVideoControls();

    // Load available videos
    await loadAvailableVideos();

    // Initialize video stream
    initVideoStream();

    // Setup idle timer
    resetIdleTimer();

    console.log('[FYP] Initialization complete');
});

// =============================================================================
// Connection Management
// =============================================================================

async function checkConnection() {
    try {
        const res = await fetch(`${API_URL}/`, { method: 'GET' });
        if (res.ok) {
            setConnectionStatus(true);
            return true;
        }
    } catch (e) {
        console.error('[FYP] Connection check failed:', e);
    }
    setConnectionStatus(false);
    return false;
}

function setConnectionStatus(connected) {
    isConnected = connected;
    const indicator = document.getElementById('connectionIndicator');
    const statusText = document.getElementById('connectionStatus');

    if (indicator) {
        if (connected) {
            indicator.className = 'w-2 h-2 rounded-full bg-green-500 shadow-[0_0_10px_#22c55e]';
        } else {
            indicator.className = 'w-2 h-2 rounded-full bg-red-500 shadow-[0_0_10px_#ef4444] animate-pulse';
        }
    }

    if (statusText) {
        statusText.textContent = connected ? 'SYS: ONLINE' : 'SYS: OFFLINE';
        statusText.className = connected
            ? 'text-[10px] text-gray-500 font-mono hidden sm:block'
            : 'text-[10px] text-red-500 font-mono hidden sm:block';
    }
}

function startConnectionMonitor() {
    if (connectionCheckInterval) clearInterval(connectionCheckInterval);
    connectionCheckInterval = setInterval(async () => {
        const wasConnected = isConnected;
        await checkConnection();

        // Reconnect stream if connection was restored
        if (!wasConnected && isConnected) {
            console.log('[FYP] Connection restored, reloading stream...');
            initVideoStream();
        }
    }, CONNECTION_CHECK_INTERVAL);
}

// =============================================================================
// Video Stream Management
// =============================================================================

function initVideoStream() {
    const videoStream = document.getElementById('videoStream');
    const loadingOverlay = document.getElementById('loadingOverlay');

    if (!videoStream) return;

    // Show loading state
    if (loadingOverlay) {
        loadingOverlay.classList.remove('hidden');
    }

    // Set up error handling
    videoStream.onerror = () => {
        console.error('[FYP] Video stream error');
        showStreamError();
    };

    videoStream.onload = () => {
        console.log('[FYP] Video stream loaded successfully');
        if (loadingOverlay) {
            loadingOverlay.classList.add('hidden');
        }
        fetchVideoInfo();
    };

    // Load stream with cache busting
    const streamUrl = `${API_URL}/video?video=${currentVideo}&t=${Date.now()}`;
    console.log('[FYP] Loading stream:', streamUrl);
    videoStream.src = streamUrl;
}

function showStreamError() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.innerHTML = `
            <div class="flex flex-col items-center gap-4">
                <iconify-icon icon="solar:danger-triangle-linear" width="48" class="text-red-500"></iconify-icon>
                <span class="text-sm text-red-400">Stream Error</span>
                <button onclick="reconnectStream()"
                    class="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-xs text-white transition-colors">
                    Retry Connection
                </button>
            </div>
        `;
        loadingOverlay.classList.remove('hidden');
    }
}

function reconnectStream() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.innerHTML = `
            <div class="flex flex-col items-center gap-3">
                <div class="w-8 h-8 border-2 border-neon border-t-transparent rounded-full animate-spin"></div>
                <span class="text-sm text-gray-400">Connecting...</span>
            </div>
        `;
    }
    setTimeout(initVideoStream, 500);
}

async function fetchVideoInfo() {
    try {
        const res = await fetch(`${API_URL}/video/info`);
        if (res.ok) {
            videoInfo = await res.json();
            console.log('[FYP] Video info:', videoInfo);
            updateLineSlider();
        }
    } catch (e) {
        console.error('[FYP] Failed to fetch video info:', e);
    }
}

// =============================================================================
// Video Controls
// =============================================================================

function changeVideo() {
    const videoSelect = document.getElementById('videoSelect');
    if (!videoSelect) return;

    currentVideo = videoSelect.value;
    console.log('[FYP] Switching to video:', currentVideo);

    // Reset stats display
    updateStatsDisplay({ in: 0, out: 0 });

    // Reload stream
    initVideoStream();
}

async function loadAvailableVideos() {
    try {
        const res = await fetch(`${API_URL}/videos`);
        if (res.ok) {
            const data = await res.json();
            console.log('[FYP] Available videos:', data.videos);

            // Dynamically populate the video select dropdown
            const videoSelect = document.getElementById('videoSelect');
            if (videoSelect && data.videos && data.videos.length > 0) {
                videoSelect.innerHTML = ''; // Clear existing options

                data.videos.forEach((video) => {
                    const option = document.createElement('option');
                    option.value = video.name;
                    option.textContent = `${video.name} (${video.size_mb} MB)`;
                    option.className = 'bg-gray-900 text-white';
                    videoSelect.appendChild(option);
                });

                // Set current video to first in list
                currentVideo = data.videos[0].name;
                console.log('[FYP] Videos loaded dynamically:', data.videos.length, 'videos');
            }
        }
    } catch (e) {
        console.error('[FYP] Failed to load videos list:', e);
    }
}

function setupVideoControls() {
    const container = document.getElementById('videoContainer');
    if (!container) return;

    container.addEventListener('mousemove', resetIdleTimer);
    container.addEventListener('click', resetIdleTimer);
    container.addEventListener('mouseenter', showControls);
    container.addEventListener('mouseleave', () => {
        resetIdleTimer();
    });
}

// =============================================================================
// Line Position Control (Percentage-based: 0-100)
// =============================================================================

function updateLineSlider() {
    const slider = document.getElementById('lineYSlider');
    if (!slider) return;

    // Slider now works directly with percentage (5-95)
    slider.min = 5;
    slider.max = 95;
    slider.value = 50;  // Default to middle
    updateLineVal(50);

    // Send initial position to backend
    updateLinePos(50);
}

function updateLineVal(percent) {
    const display = document.getElementById('lineValue');
    if (display) {
        display.innerText = `${Math.round(percent)}%`;
    }
}

async function updateLinePos(percent) {
    try {
        const response = await fetch(`${API_URL}/settings/line`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ line_percent: parseFloat(percent) }),
        });
        if (response.ok) {
            console.log(`[FYP] Line position set to ${percent}%`);
        }
    } catch (error) {
        console.error('[FYP] Error updating line:', error);
    }
}

// =============================================================================
// Stats Polling
// =============================================================================

function startStatsPolling() {
    if (statsInterval) clearInterval(statsInterval);

    statsInterval = setInterval(async () => {
        if (!isConnected) return;

        try {
            const res = await fetch(`${API_URL}/stats`);
            if (res.ok) {
                const data = await res.json();
                updateStatsDisplay(data);
            }
        } catch (e) {
            // Silent fail for stats - connection monitor will handle reconnection
        }
    }, STATS_POLL_INTERVAL);
}

function updateStatsDisplay(data) {
    const inEl = document.getElementById('inCountDisplay');
    const outEl = document.getElementById('outCountDisplay');

    if (inEl) {
        const currentIn = parseInt(inEl.innerText) || 0;
        if (data.in !== currentIn) {
            // Animate count change with pulse effect
            inEl.classList.add('scale-125');
            inEl.style.textShadow = '0 0 20px #6366f1';
            setTimeout(() => {
                inEl.classList.remove('scale-125');
                inEl.style.textShadow = '';
            }, 300);
        }
        inEl.innerText = data.in;
    }

    if (outEl) {
        const currentOut = parseInt(outEl.innerText) || 0;
        if (data.out !== currentOut) {
            // Animate count change with pulse effect
            outEl.classList.add('scale-125');
            outEl.style.textShadow = '0 0 20px #06b6d4';
            setTimeout(() => {
                outEl.classList.remove('scale-125');
                outEl.style.textShadow = '';
            }, 300);
        }
        outEl.innerText = data.out;
    }
}

// =============================================================================
// UI Visibility Controls
// =============================================================================

function showControls() {
    const controls = document.getElementById('videoControls');
    const header = document.getElementById('headerInfo');
    const stats = document.getElementById('statsOverlay');

    if (controls) {
        controls.style.opacity = '1';
        controls.style.transform = 'translateY(0)';
    }
    if (header) header.style.opacity = '1';
    if (stats) stats.style.opacity = '1';

    document.body.style.cursor = 'default';
}

function hideControls() {
    const controls = document.getElementById('videoControls');
    const header = document.getElementById('headerInfo');
    const stats = document.getElementById('statsOverlay');

    if (controls) {
        controls.style.opacity = '0';
        controls.style.transform = 'translateY(20px)';
    }
    if (header) header.style.opacity = '0';
    if (stats) stats.style.opacity = '0';

    if (document.fullscreenElement) {
        document.body.style.cursor = 'none';
    }
}

function resetIdleTimer() {
    showControls();
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(hideControls, IDLE_TIMEOUT);
}

// =============================================================================
// Fullscreen
// =============================================================================

function toggleFullscreen() {
    const container = document.getElementById('videoContainer');
    if (!container) return;

    if (!document.fullscreenElement) {
        if (container.requestFullscreen) container.requestFullscreen();
        else if (container.mozRequestFullScreen) container.mozRequestFullScreen();
        else if (container.webkitRequestFullscreen) container.webkitRequestFullscreen();
        else if (container.msRequestFullscreen) container.msRequestFullscreen();
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
    }
}

// Handle fullscreen change
document.addEventListener('fullscreenchange', () => {
    if (document.fullscreenElement) {
        resetIdleTimer();
    } else {
        showControls();
    }
});

// =============================================================================
// Reset Button
// =============================================================================

async function resetCounters() {
    try {
        await fetch(`${API_URL}/reset`, { method: 'POST' });
        updateStatsDisplay({ in: 0, out: 0 });
        console.log('[FYP] Counters reset');
    } catch (e) {
        console.error('[FYP] Failed to reset counters:', e);
    }
}
