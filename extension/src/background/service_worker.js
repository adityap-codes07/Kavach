/**
 * SmartShield — Background Service Worker (Manifest V3)
 * =====================================================
 * Handles:
 *  - Message routing between content scripts and popup
 *  - API calls to SmartShield backend
 *  - Cache management (IndexedDB for analysis history)
 *  - Context menu registration
 *  - Notification dispatch
 *  - Badge updates based on risk level
 */

import { SmartShieldAPI } from "../utils/api.js";
import { AnalysisCache } from "../utils/cache.js";

const API_BASE_URL = "https://smartshield-api.yourdomain.com/api/v1";
const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

const api = new SmartShieldAPI(API_BASE_URL);
const cache = new AnalysisCache();

// ─────────────────────────────────────────────────────────────────────────────
// Context menus
// ─────────────────────────────────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "smartshield-analyze-selection",
    title: "🛡️ Scan with SmartShield",
    contexts: ["selection"],
  });

  chrome.contextMenus.create({
    id: "smartshield-analyze-page",
    title: "🛡️ Scan this email page",
    contexts: ["page"],
    documentUrlPatterns: [
      "https://mail.google.com/*",
      "https://outlook.live.com/*",
      "https://mail.yahoo.com/*",
    ],
  });

  // Set default badge
  chrome.action.setBadgeText({ text: "" });
  chrome.action.setBadgeBackgroundColor({ color: "#22c55e" });
});

// ─────────────────────────────────────────────────────────────────────────────
// Context menu click handler
// ─────────────────────────────────────────────────────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "smartshield-analyze-selection" && info.selectionText) {
    await analyzeAndNotify(info.selectionText, tab?.id);
  }
  if (info.menuItemId === "smartshield-analyze-page" && tab?.id) {
    chrome.tabs.sendMessage(tab.id, { type: "EXTRACT_EMAIL_CONTENT" });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Keyboard shortcut handler
// ─────────────────────────────────────────────────────────────────────────────
chrome.commands.onCommand.addListener(async (command) => {
  if (command === "scan-clipboard") {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      chrome.tabs.sendMessage(tab.id, { type: "READ_CLIPBOARD_AND_SCAN" });
    }
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Message router
// ─────────────────────────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender)
    .then(sendResponse)
    .catch((err) => sendResponse({ error: err.message }));
  return true; // keep channel open for async response
});

async function handleMessage(message, sender) {
  switch (message.type) {
    case "ANALYZE_EMAIL": {
      const { content, subject, sender: emailSender, headers } = message.payload;
      return await analyzeEmail({ content, subject, sender: emailSender, headers });
    }

    case "ANALYZE_FILE": {
      const { fileData, filename } = message.payload;
      return await analyzeFile(fileData, filename);
    }

    case "GET_CACHED_RESULT": {
      const { emailHash } = message.payload;
      return await cache.get(emailHash);
    }

    case "GET_API_STATUS": {
      return await api.healthCheck();
    }

    case "GET_BENCHMARK": {
      return await api.getBenchmark();
    }

    case "CLEAR_CACHE": {
      await cache.clear();
      return { success: true };
    }

    case "EMAIL_CONTENT_EXTRACTED": {
      const tabId = sender.tab?.id;
      return await analyzeAndNotify(message.payload.content, tabId, message.payload);
    }

    default:
      throw new Error(`Unknown message type: ${message.type}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core analysis logic
// ─────────────────────────────────────────────────────────────────────────────
async function analyzeEmail({ content, subject = "", sender = "", headers = {} }) {
  if (!content?.trim()) {
    return { error: "No email content to analyze." };
  }

  // Check cache
  const hashInput = content + subject + sender;
  const cacheKey = await hashString(hashInput);
  const cached = await cache.get(cacheKey);
  if (cached) {
    return { ...cached, fromCache: true };
  }

  try {
    const result = await api.analyzeText({ content, subject, sender, headers });
    await cache.set(cacheKey, result, CACHE_TTL_MS);
    updateBadge(result.risk_score, result.risk_level);
    return result;
  } catch (err) {
    console.error("[SmartShield] Analysis failed:", err);
    return { error: `Analysis failed: ${err.message}` };
  }
}

async function analyzeFile(fileData, filename) {
  try {
    const result = await api.analyzeFile(fileData, filename);
    updateBadge(result.risk_score, result.risk_level);
    return result;
  } catch (err) {
    return { error: `File analysis failed: ${err.message}` };
  }
}

async function analyzeAndNotify(content, tabId, extras = {}) {
  const result = await analyzeEmail({ content, ...extras });

  if (result.error) return result;

  // Show browser notification for high-risk emails
  if (result.risk_score >= 60) {
    const severity = result.risk_level.toUpperCase();
    chrome.notifications.create({
      type: "basic",
      iconUrl: "assets/icons/shield-48.png",
      title: `⚠️ SmartShield: ${severity} Risk Detected`,
      message: result.recommendations?.[0]?.message || "Suspicious email detected.",
      priority: result.risk_score >= 80 ? 2 : 1,
    });
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Badge management
// ─────────────────────────────────────────────────────────────────────────────
function updateBadge(riskScore, riskLevel) {
  const colorMap = {
    safe:     "#22c55e",   // green
    low:      "#86efac",   // light green
    medium:   "#f59e0b",   // amber
    high:     "#ef4444",   // red
    critical: "#7f1d1d",   // dark red
  };

  const color = colorMap[riskLevel] || "#6b7280";
  const text = riskScore >= 20 ? String(riskScore) : "✓";

  chrome.action.setBadgeText({ text });
  chrome.action.setBadgeBackgroundColor({ color });
}

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────
async function hashString(str) {
  const buf = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(str)
  );
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 16);
}
