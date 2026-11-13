"use strict";

/* =========================================
   1. CONFIGURATION
========================================= */

const IS_LOCAL =
    location.hostname === "localhost" ||
    location.hostname === "127.0.0.1";

const CONFIG = {
    API_BASE_URL: IS_LOCAL
        ? "http://127.0.0.1:8000"
        : "https://youtube-chatbot-6kzr.onrender.com",

    ENDPOINTS: {
        health: "/health",
        loadVideo: "/api/load-video",
        loadStatus: "/api/load-status/",
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
    videoUrl: null,
    history: [],
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
        } else if (host === "youtube.com" || host === "m.youtube.com") {
            if (url.pathname === "/watch") {
                id = url.searchParams.get("v");
            } else if (["shorts", "embed", "live"].includes(parts[0])) {
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
    const { method = "GET", body = undefined } = options;

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT);

    try {
        const response = await fetch(CONFIG.API_BASE_URL + path, {
            method,
            headers: body !== undefined
                ? { "Content-Type": "application/json" }
                : undefined,
            body: body !== undefined ? JSON.stringify(body) : undefined,
            signal: controller.signal
        });

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
                "Cannot connect to the backend. The server may be waking up (wait up to 1 minute) or restarting. Please try again."
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
        button.classList.toggle("active", button.dataset.view === name);
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
        localStorage.setItem(CONFIG.HISTORY_KEY, JSON.stringify(history));
    } catch {
        // Ignore storage errors.
    }
}

function saveHistory(entry) {
    const history = getHistory().filter(item => item.id !== entry.id);
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
        $("loadBtn").classList.toggle("loading", isLoading);
    }

    if ($("loadBtnText")) {
        $("loadBtnText").textContent = isLoading ? "Loading..." : "Load Video";
    }

    if ($("loadStatus")) {
        $("loadStatus").textContent = message;
        $("loadStatus").hidden = !isLoading;
    }
}

/* =========================================
   9. LOAD YOUTUBE VIDEO
========================================= */

async function waitForVideo(videoId) {
    const startedAt = Date.now();
    let failures = 0;

    while (Date.now() - startedAt < 20 * 60 * 1000) {
        await new Promise(resolve => setTimeout(resolve, 3000));

        let status;

        try {
            status = await apiFetch(CONFIG.ENDPOINTS.loadStatus + videoId);
            failures = 0;
        } catch (error) {
            failures += 1;
            if (failures >= 6) throw error;
            continue;
        }

        if (status.state === "done") return status;

        if (status.state === "error") {
            throw new Error(status.error || "Video load nahi ho paya.");
        }

        if (status.state === "unknown") {
            throw new Error(
                "Server restart ho gaya tha. Link dobara daal ke Load Video dabao."
            );
        }

        if (status.stage) {
            setLoading(true, status.stage);
        }
    }

    throw new Error(
        "Video process hone me bahut time lag gaya. Chhoti video try karo."
    );
}

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
    state.videoId = videoId;

    setLoading(true, "Video samjha ja raha hai (pehli baar 1-2 minute lag sakte hain)...");

    try {
        let data = await apiFetch(CONFIG.ENDPOINTS.loadVideo, {
            method: "POST",
            body: {
                video_url: url,
                language: language
            }
        });

        if (data.state !== "done") {
            data = await waitForVideo(data.video_id);
        }

        if (!data.video_id) {
            throw new Error("Video ID missing from backend response.");
        }

        state.videoId = data.video_id;
        state.videoUrl = url;

        renderVideo(data.video || {});

        if ($("empty")) {
            $("empty").hidden = true;
        }

        if ($("workspace")) {
            $("workspace").hidden = false;
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

        // Summary ab alag request me load hoti hai (server par timeout na ho).
        if (data.summary) {
            renderSummary(data.summary);
            $("summaryBody").hidden = false;
            $("summaryError").hidden = true;
            $("summaryLoading").hidden = true;
        } else {
            loadSummary();
        }

    } catch (error) {
        console.error("Video loading error:", error);

        showFieldError(error.message);

    } finally {
        setLoading(false);
    }
}


/* =========================================
   11. RENDER VIDEO DETAILS
========================================= */

function renderVideo(video) {
    if ($("videoTitle")) {
        $("videoTitle").textContent = video.title || "YouTube Video";
    }

    if ($("videoChannel")) {
        $("videoChannel").textContent = video.channel || "Unknown Channel";
    }

    if ($("videoDuration")) {
        $("videoDuration").textContent = video.duration || "";
    }

    if ($("thumb")) {
        $("thumb").src =
            video.thumbnail ||
            `https://img.youtube.com/vi/${state.videoId}/hqdefault.jpg`;
        $("thumb").alt = video.title || "YouTube video thumbnail";
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
            reject(new Error("YouTube player API could not load."));
        };

        document.head.appendChild(script);
    });

    return youtubeApiPromise;
}

async function initPlayer(videoId) {
    try {
        await loadYouTubeApi();

        if (state.player && typeof state.player.loadVideoById === "function") {
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
        const tokenPattern = /(\*\*[^\*]+\*\*|__[^\_]+__|\*[^\*]+\*|_[^\_]+_|\[\d{1,2}(?::\d{2}){1,2}\])/g;

        let cursor = 0;
        let match;

        while ((match = tokenPattern.exec(text)) !== null) {
            if (match.index > cursor) {
                parent.appendChild(
                    document.createTextNode(text.slice(cursor, match.index))
                );
            }

            const token = match[0];

            if (token.startsWith("[")) {
                const label = token.slice(1, -1);
                const button = document.createElement("button");

                button.type = "button";
                button.className = "ts-link";
                button.dataset.seconds = String(timestampToSeconds(label));
                button.style.cssText = TS_STYLE;
                button.textContent = `▶ ${label}`;

                parent.appendChild(button);
                cursor = tokenPattern.lastIndex;
                continue;
            }

            const isBold = token.startsWith("**") || token.startsWith("__");
            const element = document.createElement(isBold ? "strong" : "em");

            element.textContent = isBold
                ? token.slice(2, -2)
                : token.slice(1, -1);

            parent.appendChild(element);
            cursor = tokenPattern.lastIndex;
        }

        if (cursor < text.length) {
            parent.appendChild(document.createTextNode(text.slice(cursor)));
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

            const level = Math.min(headingMatch[1].length + 1, 4);
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
            appendInline(item, (bulletMatch || numberMatch)[1]);
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
        const data = await apiFetch(CONFIG.ENDPOINTS.summary, {
            method: "POST",
            body: {
                video_id: videoId,
                language: getSelectedLanguage(),
                video_url: state.videoUrl || $("urlInput")?.value.trim() || undefined
            }
        });

        if (videoId !== state.videoId) return;

        renderSummary(data.summary || "No summary available.");
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

const TS_STYLE =
    "background:#e8f0fe;color:#1a56db;border:0;border-radius:999px;" +
    "padding:1px 9px;margin:0 2px;font-size:.85em;font-weight:600;cursor:pointer;";

function timestampToSeconds(label) {
    const parts = label.split(":").map(Number);
    return parts.length === 3
        ? parts[0] * 3600 + parts[1] * 60 + parts[2]
        : parts[0] * 60 + parts[1];
}

function formatAnswer(answer) {
    return escapeHtml(answer)
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(
            /\[(\d{1,2}(?::\d{2}){1,2})\]/g,
            (match, label) =>
                `<button type="button" class="ts-link" ` +
                `data-seconds="${timestampToSeconds(label)}" ` +
                `style="${TS_STYLE}" title="Video is time par chalao">▶ ${label}</button>`
        )
        .replace(/\n/g, "<br>");
}

function seekVideo(seconds) {
    if (!state.player || typeof state.player.seekTo !== "function") {
        toast("Video player abhi ready nahi hai.", "error");
        return;
    }

    state.player.seekTo(seconds, true);

    if (typeof state.player.playVideo === "function") {
        state.player.playVideo();
    }

    document.querySelector(".player-wrap")?.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });
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
    state.history = [];

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
    addMessage("user", escapeHtml(question).replace(/\n/g, "<br>"));

    $("chatInput").value = "";
    autoGrow();

    state.busy = true;
    $("sendBtn").disabled = true;

    const typing = addMessage(
        "ai",
        '<div class="typing"><i></i><i></i><i></i></div>'
    );

    try {
        const history = state.history.slice(-8);

        const data = await apiFetch(CONFIG.ENDPOINTS.chat, {
            method: "POST",
            body: {
                video_id: state.videoId,
                question: question,
                language: getSelectedLanguage(),
                video_url: state.videoUrl || $("urlInput")?.value.trim() || undefined,
                history: history
            }
        });

        typing?.remove();

        const answerText = data.answer || "No answer received.";

        state.history.push({ role: "user", content: question });
        state.history.push({ role: "assistant", content: answerText.slice(0, 1500) });

        addMessage("ai", formatAnswer(answerText));

    } catch (error) {
        typing?.remove();

        let message = error.message;

        // Render restart hone par server memory se video hat jata hai.
        if (message.toLowerCase().includes("not loaded")) {
            message =
                "Server restart hua tha, isliye video memory se hat gaya. " +
                "Upar video link dobara 'Load Video' karo, phir sawal pucho.";
        }

        addMessage("ai error", escapeHtml(message));
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
    input.style.height = Math.min(input.scrollHeight, 120) + "px";
}


/* =========================================
   18. INITIALIZE APPLICATION
========================================= */

function init() {
    // Theme
    initTheme();

    // Click on any [MM:SS] button (chat or summary) -> jump the video there.
    document.addEventListener("click", event => {
        const button = event.target.closest(".ts-link");
        if (!button) return;

        const seconds = Number(button.dataset.seconds);
        if (Number.isFinite(seconds)) {
            seekVideo(seconds);
        }
    });

    $("themeToggle")?.addEventListener("click", () => {
        const current = document.documentElement.dataset.theme;
        applyTheme(current === "dark" ? "light" : "dark");
    });

    // Navigation
    $("nav")?.addEventListener("click", event => {
        const button = event.target.closest(".nav-btn");
        if (button) {
            setView(button.dataset.view);
        }
    });

    // Video form
    $("videoForm")?.addEventListener("submit", handleLoad);

    // Chat form
    $("chatForm")?.addEventListener("submit", handleSend);

    // Clear chat
    $("clearChat")?.addEventListener("click", resetChat);

    // Language change
    $("languageSelect")?.addEventListener("change", () => {
        if (state.videoId) {
            loadSummary();
        }
    });

    // Chat input
    $("chatInput")?.addEventListener("input", autoGrow);

    // Enter to send
    $("chatInput")?.addEventListener("keydown", event => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            $("chatForm")?.requestSubmit();
        }
    });

    // Clear history
    $("clearHistory")?.addEventListener("click", () => {
        setHistory([]);
        renderHistory();
        toast("History cleared.");
    });

    // Load video from history
    $("historyList")?.addEventListener("click", event => {
        const button = event.target.closest("[data-url]");
        if (!button) return;

        $("urlInput").value = button.dataset.url;
        setView("home");
        $("videoForm")?.requestSubmit();
    });

    // Backend health check (free Render server so raha ho to jagane ke liye)
    apiFetch(CONFIG.ENDPOINTS.health)
        .then(() => {
            console.log("FastAPI backend connected successfully.");
        })
        .catch(error => {
            console.error("Backend connection failed:", error);
            toast("Server is waking up, please wait up to 1 minute...", "error");
        });
}


/* =========================================
   19. START APPLICATION
========================================= */

document.addEventListener("DOMContentLoaded", init);