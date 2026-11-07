
"use strict";

/* =========================================
   1. CONFIGURATION
========================================= */

const CONFIG = {
    API_BASE_URL: "http://127.0.0.1:8000",

    ENDPOINTS: {
        health: "/health",
        loadVideo: "/api/load-video",
        manualTranscript: "/api/manual-transcript",
        summary: "/api/summary",
        chat: "/api/chat"
    },

    REQUEST_TIMEOUT: 300000,
    HISTORY_KEY: "ytchat_history",
    THEME_KEY: "ytchat_theme",
    HISTORY_LIMIT: 20
};

const $ = (id) => document.getElementById(id);

const state = {
    videoId: null,
    player: null,
    busy: false
};


/* =========================================
   2. YOUTUBE VIDEO ID
========================================= */

function extractVideoId(raw) {
    try {
        const url = new URL(raw.trim());
        const host = url.hostname.replace(/^www\./, "");
        const parts = url.pathname.split("/").filter(Boolean);

        let id = null;

        if (host === "youtu.be") {
            id = parts[0];
        } else if (
            host === "youtube.com" ||
            host === "m.youtube.com"
        ) {
            if (url.pathname === "/watch") {
                id = url.searchParams.get("v");
            } else if (
                ["shorts", "embed", "live"].includes(parts[0])
            ) {
                id = parts[1];
            }
        }

        return id && /^[\w-]{11}$/.test(id) ? id : null;

    } catch {
        return null;
    }
}


/* =========================================
   3. UTILITY FUNCTIONS
========================================= */

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    })[char]);
}

function getSelectedLanguage() {
    return $("languageSelect")?.value || "English";
}

let toastTimer;

function toast(message, type = "info") {
    const element = $("toast");

    if (!element) {
        console.log(message);
        return;
    }

    element.textContent = message;
    element.className = type === "error" ? "toast error" : "toast";
    element.hidden = false;

    clearTimeout(toastTimer);

    toastTimer = setTimeout(() => {
        element.hidden = true;
    }, 4500);
}

function showFieldError(message) {
    const element = $("urlError");

    if (!element) {
        if (message) toast(message, "error");
        return;
    }

    element.textContent = message || "";
    element.hidden = !message;
}


/* =========================================
   4. API FETCH FUNCTION
========================================= */

async function apiFetch(path, options = {}) {
    const {
        method = "GET",
        body = undefined
    } = options;

    const controller = new AbortController();

    const timer = setTimeout(() => {
        controller.abort();
    }, CONFIG.REQUEST_TIMEOUT);

    try {
        const response = await fetch(
            CONFIG.API_BASE_URL + path,
            {
                method,

                headers: body !== undefined
                    ? { "Content-Type": "application/json" }
                    : undefined,

                body: body !== undefined
                    ? JSON.stringify(body)
                    : undefined,

                signal: controller.signal
            }
        );

        let data = null;

        try {
            data = await response.json();
        } catch {
            // Invalid JSON response.
        }

        if (!response.ok) {
            const detail = data?.detail;

            let message = `Request failed: ${response.status}`;

            if (typeof detail === "string") {
                message = detail;

            } else if (Array.isArray(detail)) {
                message = detail
                    .map(item => item.msg || JSON.stringify(item))
                    .join(", ");
            }

            throw new Error(message);
        }

        return data || {};

    } catch (error) {
        if (error.name === "AbortError") {
            throw new Error("Request timed out. Please try again.");
        }

        if (error instanceof TypeError) {
            throw new Error(
                "Cannot connect to the backend. Please check the server and internet connection."
            );
        }

        throw error;

    } finally {
        clearTimeout(timer);
    }
}


/* =========================================
   5. NAVIGATION
========================================= */

function setView(name) {

    document.querySelectorAll(".view").forEach(view => {

        const isActive = view.id === `view-${name}`;

        view.classList.toggle("active", isActive);
        view.hidden = !isActive;

    });

    document.querySelectorAll(".nav-btn").forEach(button => {

        button.classList.toggle(
            "active",
            button.dataset.view === name
        );

    });

    if (name === "history") {
        renderHistory();
    }
}


/* =========================================
   6. DARK / LIGHT THEME
========================================= */

function applyTheme(theme) {

    document.documentElement.dataset.theme = theme;

    const button = $("themeToggle");

    if (button) {
        button.textContent = theme === "dark" ? "☀️" : "🌙";
    }

    try {
        localStorage.setItem(CONFIG.THEME_KEY, theme);
    } catch {
        // Ignore storage errors.
    }
}

function initTheme() {

    let savedTheme = null;

    try {
        savedTheme = localStorage.getItem(CONFIG.THEME_KEY);
    } catch {
        // Ignore storage errors.
    }

    const preferredTheme =
        window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark"
            : "light";

    applyTheme(savedTheme || preferredTheme);
}


/* =========================================
   7. VIDEO HISTORY
========================================= */

function getHistory() {

    try {

        const history = JSON.parse(
            localStorage.getItem(CONFIG.HISTORY_KEY) || "[]"
        );

        return Array.isArray(history) ? history : [];

    } catch {
        return [];
    }
}

function setHistory(history) {

    try {

        localStorage.setItem(
            CONFIG.HISTORY_KEY,
            JSON.stringify(history)
        );

    } catch {
        // Ignore storage errors.
    }
}

function saveHistory(entry) {

    const history = getHistory().filter(
        item => item.id !== entry.id
    );

    history.unshift(entry);

    setHistory(history.slice(0, CONFIG.HISTORY_LIMIT));
}

function renderHistory() {

    const container = $("historyList");

    if (!container) return;

    const history = getHistory();

    if (!history.length) {

        container.innerHTML =
            '<p class="muted">No videos yet. Load a video from Home.</p>';

        return;
    }

    container.innerHTML = history.map(item => `

        <div class="hist">

            <div>
                <strong>${escapeHtml(item.title)}</strong>

                <span class="muted">
                    ${escapeHtml(item.channel)}
                    · ${escapeHtml(new Date(item.at).toLocaleString())}
                </span>
            </div>

            <button
                class="btn ghost small"
                type="button"
                data-url="${escapeHtml(item.url)}">
                Load
            </button>

        </div>

    `).join("");
}


/* =========================================
   8. LOADING STATE
========================================= */

function setLoading(isLoading, message = "") {

    if ($("loadBtn")) {

        $("loadBtn").disabled = isLoading;

        $("loadBtn").classList.toggle(
            "loading",
            isLoading
        );
    }

    if ($("loadBtnText")) {

        $("loadBtnText").textContent =
            isLoading ? "Loading..." : "Load Video";
    }

    if ($("loadStatus")) {

        $("loadStatus").textContent = message;

        $("loadStatus").hidden = !isLoading;
    }
}

function setManualTranscriptVisible(visible) {

    const box = $("manualTranscriptBox");

    if (box) {
        box.hidden = !visible;
    }
}


/* =========================================
   9. LOAD YOUTUBE VIDEO
========================================= */

async function handleLoad(event) {

    event.preventDefault();

    const url = $("urlInput")?.value.trim();

    const language = getSelectedLanguage();

    if (!url) {

        showFieldError("Please paste a YouTube URL.");

        return;
    }

    const videoId = extractVideoId(url);

    if (!videoId) {

        showFieldError("Invalid YouTube URL.");

        return;
    }

    showFieldError("");

    setManualTranscriptVisible(false);

    state.videoId = videoId;

    setLoading(
        true,
        "Analyzing video and preparing AI features..."
    );

    try {

        const data = await apiFetch(
            CONFIG.ENDPOINTS.loadVideo,
            {
                method: "POST",

                body: {
                    video_url: url,
                    language: language
                }
            }
        );

        if (!data.video_id) {

            throw new Error(
                "Video ID missing from backend response."
            );
        }

        state.videoId = data.video_id;

        renderVideo(data.video || {});

        if ($("empty")) {
            $("empty").hidden = true;
        }

        if ($("workspace")) {
            $("workspace").hidden = false;
        }

        if (data.summary) {

            renderSummary(data.summary);

            $("summaryBody").hidden = false;
            $("summaryError").hidden = true;
            $("summaryLoading").hidden = true;

        } else {

            $("summaryBody").hidden = true;
            $("summaryLoading").hidden = true;

            if ($("summaryError")) {

                $("summaryError").textContent =
                    data.summary_error || "Summary unavailable.";

                $("summaryError").hidden = false;
            }
        }

        saveHistory({
            id: state.videoId,
            url: url,
            title: data.video?.title || "YouTube Video",
            channel: data.video?.channel || "",
            at: Date.now()
        });

        resetChat();

        toast("Video loaded successfully!");

    } catch (error) {

        console.error("Video loading error:", error);

        state.videoId = videoId;

        setManualTranscriptVisible(true);

        showFieldError(
            `${error.message} You can paste the transcript manually below.`
        );

    } finally {

        setLoading(false);

    }
}


/* =========================================
   10. MANUAL TRANSCRIPT SUBMISSION
========================================= */

async function handleManualTranscript(event) {

    event.preventDefault();

    const transcript =
        $("manualTranscriptInput")?.value.trim();

    if (!state.videoId) {

        toast(
            "Please enter and load a valid YouTube video URL first.",
            "error"
        );

        return;
    }

    if (!transcript || transcript.length < 20) {

        toast(
            "Please paste a valid transcript (at least 20 characters).",
            "error"
        );

        return;
    }

    const button = $("manualTranscriptBtn");

    if (button) {
        button.disabled = true;
        button.textContent = "Processing...";
    }

    try {

        const data = await apiFetch(
            CONFIG.ENDPOINTS.manualTranscript,
            {
                method: "POST",

                body: {
                    video_id: state.videoId,
                    transcript: transcript,
                    language: getSelectedLanguage()
                }
            }
        );

        if (data.video) {

            renderVideo(data.video);

        } else {

            renderVideo({
                title: "YouTube Video",
                channel: "YouTube",
                thumbnail:
                    `https://img.youtube.com/vi/${state.videoId}/hqdefault.jpg`
            });
        }

        if ($("empty")) {
            $("empty").hidden = true;
        }

        if ($("workspace")) {
            $("workspace").hidden = false;
        }

        const url =
            $("urlInput")?.value.trim() ||
            `https://www.youtube.com/watch?v=${state.videoId}`;

        saveHistory({
            id: state.videoId,
            url: url,
            title: data.video?.title || "YouTube Video",
            channel: data.video?.channel || "",
            at: Date.now()
        });

        setManualTranscriptVisible(false);

        resetChat();

        if ($("summaryLoading")) {
            $("summaryLoading").hidden = false;
        }

        if ($("summaryBody")) {
            $("summaryBody").hidden = true;
        }

        if ($("summaryError")) {
            $("summaryError").hidden = true;
        }

        toast(
            `Transcript processed successfully (${data.transcript_segments || 0} segments).`
        );

        await loadSummary();

    } catch (error) {

        console.error("Manual transcript error:", error);

        toast(error.message, "error");

    } finally {

        if (button) {
            button.disabled = false;
            button.textContent = "Process Transcript";
        }

    }
}


/* =========================================
   11. RENDER VIDEO DETAILS
========================================= */

function renderVideo(video) {

    if ($("videoTitle")) {

        $("videoTitle").textContent =
            video.title || "YouTube Video";
    }

    if ($("videoChannel")) {

        $("videoChannel").textContent =
            video.channel || "Unknown Channel";
    }

    if ($("videoDuration")) {

        $("videoDuration").textContent =
            video.duration || "";
    }

    if ($("thumb")) {

        $("thumb").src =
            video.thumbnail ||
            `https://img.youtube.com/vi/${state.videoId}/hqdefault.jpg`;

        $("thumb").alt =
            video.title || "YouTube video thumbnail";
    }

    initPlayer(state.videoId);
}


/* =========================================
   12. YOUTUBE PLAYER
========================================= */

let youtubeApiPromise = null;

function loadYouTubeApi() {

    if (window.YT && window.YT.Player) {
        return Promise.resolve();
    }

    if (youtubeApiPromise) {
        return youtubeApiPromise;
    }

    youtubeApiPromise = new Promise((resolve, reject) => {

        window.onYouTubeIframeAPIReady = resolve;

        const script = document.createElement("script");

        script.src = "https://www.youtube.com/iframe_api";

        script.onerror = () => {

            youtubeApiPromise = null;

            reject(
                new Error("YouTube player API could not load.")
            );
        };

        document.head.appendChild(script);

    });

    return youtubeApiPromise;
}

async function initPlayer(videoId) {

    try {

        await loadYouTubeApi();

        if (
            state.player &&
            typeof state.player.loadVideoById === "function"
        ) {

            state.player.loadVideoById(videoId);

            return;
        }

        if (!$("player")) return;

        state.player = new YT.Player("player", {

            videoId: videoId,

            width: "100%",

            height: "100%",

            playerVars: {
                rel: 0,
                playsinline: 1
            }

        });

    } catch (error) {

        console.error("YouTube player error:", error);

        toast(error.message, "error");

    }
}


/* =========================================
   13. SAFE MARKDOWN SUMMARY RENDERER
========================================= */

function renderSummary(markdown) {

    const container = $("summaryText");

    if (!container) return;

    container.replaceChildren();

    let source = String(markdown || "");

    source = source
        .replace(/\\?<br\s*\/?>/gi, "\n")
        .replace(/\\([*_#|])/g, "$1")
        .replace(/\r\n?/g, "\n");

    const lines = source.split("\n");

    let activeList = null;
    let activeListType = null;

    function closeList() {

        activeList = null;
        activeListType = null;

    }

    function appendInline(parent, value) {

        const text = String(value || "");

        const tokenPattern =
            /(\*\*[^\*]+\*\*|__[^\_]+__|\*[^\*]+\*|_[^\_]+_)/g;

        let cursor = 0;
        let match;

        while ((match = tokenPattern.exec(text)) !== null) {

            if (match.index > cursor) {

                parent.appendChild(
                    document.createTextNode(
                        text.slice(cursor, match.index)
                    )
                );
            }

            const token = match[0];

            const isBold =
                token.startsWith("**") ||
                token.startsWith("__");

            const element = document.createElement(
                isBold ? "strong" : "em"
            );

            element.textContent = isBold
                ? token.slice(2, -2)
                : token.slice(1, -1);

            parent.appendChild(element);

            cursor = tokenPattern.lastIndex;

        }

        if (cursor < text.length) {

            parent.appendChild(
                document.createTextNode(
                    text.slice(cursor)
                )
            );

        }
    }

    for (const rawLine of lines) {

        const line = rawLine.trim();

        if (!line || /^[-*_]{3,}$/.test(line)) {

            closeList();

            continue;
        }

        // Markdown table fallback.
        if (line.startsWith("|") && line.endsWith("|")) {

            if (/^\|[\s:|-]+\|$/.test(line)) {
                continue;
            }

            closeList();

            const cells = line
                .split("|")
                .slice(1, -1)
                .map(cell => cell.trim())
                .filter(Boolean);

            if (cells.length) {

                const paragraph = document.createElement("p");

                paragraph.className = "summary-table-fallback";

                appendInline(paragraph, cells.join(" — "));

                container.appendChild(paragraph);

            }

            continue;
        }

        // Headings.
        const headingMatch = line.match(/^(#{1,4})\s+(.+)$/);

        if (headingMatch) {

            closeList();

            const level = Math.min(
                headingMatch[1].length + 1,
                4
            );

            const heading = document.createElement(`h${level}`);

            appendInline(heading, headingMatch[2]);

            container.appendChild(heading);

            continue;
        }

        // Bullet points.
        const bulletMatch = line.match(/^[-*•]\s+(.+)$/);

        // Numbered points.
        const numberMatch = line.match(/^\d+[.)]\s+(.+)$/);

        if (bulletMatch || numberMatch) {

            const listType = bulletMatch ? "ul" : "ol";

            if (!activeList || activeListType !== listType) {

                closeList();

                activeList = document.createElement(listType);

                activeListType = listType;

                container.appendChild(activeList);

            }

            const item = document.createElement("li");

            appendInline(
                item,
                (bulletMatch || numberMatch)[1]
            );

            activeList.appendChild(item);

            continue;
        }

        // Normal paragraph.
        closeList();

        const paragraph = document.createElement("p");

        appendInline(paragraph, line);

        container.appendChild(paragraph);

    }
}


/* =========================================
   14. VIDEO SUMMARY
========================================= */

async function loadSummary() {

    if (!state.videoId) {

        toast("Load a video first.", "error");

        return;
    }

    $("summaryLoading").hidden = false;
    $("summaryBody").hidden = true;
    $("summaryError").hidden = true;

    const videoId = state.videoId;

    try {

        const data = await apiFetch(
            CONFIG.ENDPOINTS.summary,
            {
                method: "POST",

                body: {
                    video_id: videoId,
                    language: getSelectedLanguage()
                }
            }
        );

        if (videoId !== state.videoId) return;

        renderSummary(
            data.summary || "No summary available."
        );

        $("summaryBody").hidden = false;

    } catch (error) {

        $("summaryError").textContent = error.message;

        $("summaryError").hidden = false;

    } finally {

        $("summaryLoading").hidden = true;

    }
}


/* =========================================
   15. CHAT MESSAGES
========================================= */

function formatAnswer(answer) {

    return escapeHtml(answer)

        .replace(
            /\*\*(.+?)\*\*/g,
            "<strong>$1</strong>"
        )

        .replace(/\n/g, "<br>");

}

function scrollChat() {

    const container = $("chatMessages");

    if (!container) return;

    container.scrollTo({
        top: container.scrollHeight,
        behavior: "smooth"
    });

}

function addMessage(role, html) {

    const container = $("chatMessages");

    if (!container) return null;

    const message = document.createElement("div");

    message.className = `msg ${role}`;

    message.innerHTML = `

        <div class="avatar">
            ${role === "user" ? "Y" : "AI"}
        </div>

        <div class="bubble">
            ${html}
        </div>

    `;

    container.appendChild(message);

    scrollChat();

    return message;
}

function resetChat() {

    const container = $("chatMessages");

    if (!container) return;

    container.innerHTML = "";

    addMessage(
        "ai",
        "Hi! Ask me anything about this video. I will answer using the video content."
    );

}


/* =========================================
   16. ASK QUESTION
========================================= */

async function handleSend(event) {

    event.preventDefault();

    const question = $("chatInput")?.value.trim();

    if (!question || state.busy) return;

    if (!state.videoId) {

        toast("Please load a video first.", "error");

        return;
    }

    // Show user message.
    addMessage(
        "user",
        escapeHtml(question).replace(/\n/g, "<br>")
    );

    $("chatInput").value = "";

    autoGrow();

    state.busy = true;

    $("sendBtn").disabled = true;

    const typing = addMessage(
        "ai",
        '<div class="typing"><i></i><i></i><i></i></div>'
    );

    try {

        const data = await apiFetch(
            CONFIG.ENDPOINTS.chat,
            {
                method: "POST",

                body: {
                    video_id: state.videoId,
                    question: question,
                    language: getSelectedLanguage()
                }
            }
        );

        typing?.remove();

        addMessage(
            "ai",
            formatAnswer(data.answer || "No answer received.")
        );

    } catch (error) {

        typing?.remove();

        addMessage(
            "ai error",
            escapeHtml(error.message)
        );

        console.error("Chat error:", error);

    } finally {

        state.busy = false;

        $("sendBtn").disabled = false;

        $("chatInput").focus();

    }
}


/* =========================================
   17. CHAT INPUT RESIZE
========================================= */

function autoGrow() {

    const input = $("chatInput");

    if (!input) return;

    input.style.height = "auto";

    input.style.height =
        Math.min(input.scrollHeight, 120) + "px";

}


/* =========================================
   18. INITIALIZE APPLICATION
========================================= */

function init() {

    // Theme
    initTheme();

    $("themeToggle")?.addEventListener("click", () => {

        const current =
            document.documentElement.dataset.theme;

        applyTheme(
            current === "dark" ? "light" : "dark"
        );

    });

    // Navigation
    $("nav")?.addEventListener("click", event => {

        const button = event.target.closest(".nav-btn");

        if (button) {
            setView(button.dataset.view);
        }

    });

    // Video form
    $("videoForm")?.addEventListener(
        "submit",
        handleLoad
    );

    // Manual transcript form
    $("manualTranscriptForm")?.addEventListener(
        "submit",
        handleManualTranscript
    );

    // Chat form
    $("chatForm")?.addEventListener(
        "submit",
        handleSend
    );

    // Clear chat
    $("clearChat")?.addEventListener(
        "click",
        resetChat
    );

    // Language change
    $("languageSelect")?.addEventListener(
        "change",
        () => {

            if (state.videoId) {
                loadSummary();
            }

        }
    );

    // Chat input
    $("chatInput")?.addEventListener(
        "input",
        autoGrow
    );

    // Enter to send
    $("chatInput")?.addEventListener(
        "keydown",
        event => {

            if (
                event.key === "Enter" &&
                !event.shiftKey
            ) {

                event.preventDefault();

                $("chatForm")?.requestSubmit();

            }

        }
    );

    // Clear history
    $("clearHistory")?.addEventListener(
        "click",
        () => {

            setHistory([]);

            renderHistory();

            toast("History cleared.");

        }
    );

    // Load video from history
    $("historyList")?.addEventListener(
        "click",
        event => {

            const button =
                event.target.closest("[data-url]");

            if (!button) return;

            $("urlInput").value = button.dataset.url;

            setView("home");

            $("videoForm")?.requestSubmit();

        }
    );

    // Backend health check
    apiFetch(CONFIG.ENDPOINTS.health)

        .then(() => {

            console.log(
                "FastAPI backend connected successfully."
            );

        })

        .catch(error => {

            console.error(
                "Backend connection failed:",
                error
            );

            toast(error.message, "error");

        });

}


/* =========================================
   19. START APPLICATION
========================================= */

document.addEventListener(
    "DOMContentLoaded",
    init
);
