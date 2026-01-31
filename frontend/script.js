const API_URL = 'http://127.0.0.1:8000';
let statsInterval = null;
let idleTimer = null;
const IDLE_TIMEOUT = 5000; // 5 seconds hide

document.addEventListener('DOMContentLoaded', () => {
    // Start stats polling
    startStatsPolling();

    // Setup user activity listeners for the video container
    const container = document.getElementById('videoContainer');
    if (container) {
        container.addEventListener('mousemove', resetIdleTimer);
        container.addEventListener('click', resetIdleTimer);
        // Force show on hover start
        container.addEventListener('mouseenter', showControls);
        // Hide on leave (optional, or keep 5s persistence)
        container.addEventListener('mouseleave', () => {
             // We can either hide immediately or wait for timer.
             // User asked for "disappear if mouse stays non-active", implies timer.
             resetIdleTimer();
        });
    }

    resetIdleTimer();
});

function changeVideo() {
    const videoSelect = document.getElementById('videoSelect');
    const videoStream = document.getElementById('videoStream');
    if (videoSelect && videoStream) {
        const selectedVideo = videoSelect.value;
        videoStream.src = `${API_URL}/video?video=${selectedVideo}&t=${Date.now()}`;
    }
}

function updateLineVal(val) {
    const display = document.getElementById('lineValue');
    if (display) {
        display.innerText = `${val}px`;
    }
}

async function updateLinePos(val) {
    try {
        await fetch(`${API_URL}/settings/line`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ line_y: parseInt(val) }),
        });
    } catch (error) {
        console.error('Error updating line:', error);
    }
}

function startStatsPolling() {
    if (statsInterval) clearInterval(statsInterval);
    statsInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/stats`);
            if (res.ok) {
                const data = await res.json();
                const inEl = document.getElementById('inCountDisplay');
                const outEl = document.getElementById('outCountDisplay');
                if (inEl) inEl.innerText = data.in;
                if (outEl) outEl.innerText = data.out;
            }
        } catch (e) {
            console.error(e);
        }
    }, 1000);
}

// UI Interaction
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

    // Optional: hide cursor in fullscreen
    if (document.fullscreenElement) {
        document.body.style.cursor = 'none';
    }
}

function resetIdleTimer() {
    showControls();
    if (idleTimer) clearTimeout(idleTimer);
    idleTimer = setTimeout(hideControls, IDLE_TIMEOUT);
}

// Fullscreen
function toggleFullscreen() {
    const container = document.getElementById('videoContainer');
    if (!document.fullscreenElement) {
        if (container.requestFullscreen) container.requestFullscreen();
        else if (container.mozRequestFullScreen) container.mozRequestFullScreen();
        else if (container.webkitRequestFullscreen) container.webkitRequestFullscreen();
        else if (container.msRequestFullscreen) container.msRequestFullscreen();
    } else {
        if (document.exitFullscreen) document.exitFullscreen();
    }
}
