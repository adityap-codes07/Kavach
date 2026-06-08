/**
 * SmartShield — Content Script
 * ==============================
 * Injected into: Gmail, Outlook, Yahoo Mail, ProtonMail
 *
 * Features:
 *  - Auto-detect open email in reading pane
 *  - Extract subject, sender, headers, and body text
 *  - Inject SmartShield inline risk badge into email header area
 *  - One-click "Scan with SmartShield" button per email
 *  - Passive background scanning with configurable threshold
 */

(function () {
  "use strict";

  if (window.__smartshield_injected) return;
  window.__smartshield_injected = true;

  // ──────────────────────────────────────────────────────────────────────────
  // Platform detection
  // ──────────────────────────────────────────────────────────────────────────
  const PLATFORM = detectPlatform();

  function detectPlatform() {
    const host = window.location.hostname;
    if (host.includes("mail.google.com"))    return "gmail";
    if (host.includes("outlook"))            return "outlook";
    if (host.includes("mail.yahoo.com"))     return "yahoo";
    if (host.includes("mail.proton.me"))     return "proton";
    return "unknown";
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Platform-specific selectors
  // ──────────────────────────────────────────────────────────────────────────
  const SELECTORS = {
    gmail: {
      emailContainer: ".ii.gt",
      subject:        "h2.hP",
      sender:         ".gD",
      headerArea:     ".ade",
      body:           ".ii.gt .a3s",
    },
    outlook: {
      emailContainer: "[data-app-section='ReadingPane']",
      subject:        "[data-tid='subject']",
      sender:         "[data-tid='SenderName']",
      headerArea:     "[data-tid='MessageHeader']",
      body:           "[data-tid='message-body']",
    },
    yahoo: {
      emailContainer: "#message-view-body",
      subject:        "[data-test-id='message-subject']",
      sender:         "[data-test-id='display-name']",
      headerArea:     "[data-test-id='message-header']",
      body:           "[data-test-id='message-body']",
    },
    proton: {
      emailContainer: ".message-container",
      subject:        ".subject",
      sender:         ".addressContent",
      headerArea:     ".message-header",
      body:           ".message-content",
    },
  };

  const sel = SELECTORS[PLATFORM] || SELECTORS.gmail;

  // ──────────────────────────────────────────────────────────────────────────
  // Email extraction
  // ──────────────────────────────────────────────────────────────────────────
  function extractEmailData() {
    const subject = document.querySelector(sel.subject)?.textContent?.trim() || "";
    const sender  = document.querySelector(sel.sender)?.textContent?.trim()  || "";
    const bodyEl  = document.querySelector(sel.body);
    const body    = bodyEl ? getTextContent(bodyEl) : "";

    return { subject, sender, content: `Subject: ${subject}\n\n${body}` };
  }

  function getTextContent(el) {
    // Prefer innerText for rendered text, fall back to textContent
    const text = el.innerText || el.textContent || "";
    // Collapse excessive whitespace
    return text.replace(/\s{3,}/g, "\n\n").trim().slice(0, 10_000);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Risk badge injection
  // ──────────────────────────────────────────────────────────────────────────
  const BADGE_ID = "smartshield-badge";

  function injectBadge(riskScore, riskLevel, classification) {
    removeBadge();

    const headerArea = document.querySelector(sel.headerArea);
    if (!headerArea) return;

    const colorMap = {
      safe:     { bg: "#dcfce7", text: "#166534", border: "#86efac" },
      low:      { bg: "#f0fdf4", text: "#15803d", border: "#4ade80" },
      medium:   { bg: "#fefce8", text: "#854d0e", border: "#fbbf24" },
      high:     { bg: "#fef2f2", text: "#991b1b", border: "#f87171" },
      critical: { bg: "#fee2e2", text: "#7f1d1d", border: "#ef4444" },
    };
    const colors = colorMap[riskLevel] || colorMap.medium;

    const badge = document.createElement("div");
    badge.id = BADGE_ID;
    badge.style.cssText = `
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 12px;
      margin: 8px 0;
      background: ${colors.bg};
      border: 1px solid ${colors.border};
      border-radius: 6px;
      font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif;
      font-size: 12px;
      font-weight: 500;
      color: ${colors.text};
      cursor: pointer;
      user-select: none;
      z-index: 9999;
      transition: opacity 0.2s;
    `;

    const icon = riskLevel === "safe" ? "✅" :
                 riskLevel === "low"  ? "🟡" :
                 riskLevel === "critical" ? "🚨" : "⚠️";

    badge.innerHTML = `
      <span>${icon} SmartShield</span>
      <span style="font-weight: 700;">Risk: ${riskScore}/100</span>
      <span style="opacity: 0.7;">${classification}</span>
      <span style="opacity: 0.5; font-size: 10px;">↗ Details</span>
    `;

    badge.addEventListener("click", () => {
      chrome.runtime.sendMessage({ type: "OPEN_POPUP" });
    });

    headerArea.prepend(badge);
  }

  function removeBadge() {
    document.getElementById(BADGE_ID)?.remove();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Scan button injection
  // ──────────────────────────────────────────────────────────────────────────
  const SCAN_BTN_ID = "smartshield-scan-btn";

  function injectScanButton() {
    if (document.getElementById(SCAN_BTN_ID)) return;

    const headerArea = document.querySelector(sel.headerArea);
    if (!headerArea) return;

    const btn = document.createElement("button");
    btn.id = SCAN_BTN_ID;
    btn.style.cssText = `
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 12px;
      background: #1d4ed8;
      color: white;
      border: none;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 500;
      font-family: 'Google Sans', system-ui, sans-serif;
      cursor: pointer;
      transition: background 0.15s;
    `;
    btn.innerHTML = "🛡️ Scan";
    btn.title = "Scan with SmartShield";

    btn.addEventListener("mouseenter", () => btn.style.background = "#1e40af");
    btn.addEventListener("mouseleave", () => btn.style.background = "#1d4ed8");

    btn.addEventListener("click", async () => {
      btn.textContent = "⏳ Scanning…";
      btn.disabled = true;
      const data = extractEmailData();
      const result = await chrome.runtime.sendMessage({
        type: "ANALYZE_EMAIL",
        payload: data,
      });
      btn.textContent = "🛡️ Scan";
      btn.disabled = false;
      if (result && !result.error) {
        injectBadge(result.risk_score, result.risk_level, result.classification);
        // Store result for popup
        chrome.storage.session.set({ lastResult: result });
      }
    });

    headerArea.appendChild(btn);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Message listeners (from background / popup)
  // ──────────────────────────────────────────────────────────────────────────
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "EXTRACT_EMAIL_CONTENT") {
      sendResponse(extractEmailData());
    }

    if (message.type === "READ_CLIPBOARD_AND_SCAN") {
      navigator.clipboard.readText().then(text => {
        sendResponse({ content: text });
        chrome.runtime.sendMessage({
          type: "ANALYZE_EMAIL",
          payload: { content: text, subject: "", sender: "" },
        });
      }).catch(() => sendResponse({ error: "Clipboard access denied" }));
      return true;
    }

    if (message.type === "SHOW_RESULT") {
      const { risk_score, risk_level, classification } = message.payload;
      injectBadge(risk_score, risk_level, classification);
    }
  });

  // ──────────────────────────────────────────────────────────────────────────
  // MutationObserver — re-inject button when email changes
  // ──────────────────────────────────────────────────────────────────────────
  let debounceTimer;
  const observer = new MutationObserver(() => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      // Check if email reading pane is open
      const container = document.querySelector(sel.emailContainer);
      if (container && !document.getElementById(SCAN_BTN_ID)) {
        injectScanButton();
        removeBadge();
      }
    }, 500);
  });

  observer.observe(document.body, {
    childList: true,
    subtree: true,
  });

  // Initial injection attempt
  setTimeout(injectScanButton, 2000);

  console.log(`[SmartShield] Content script loaded on ${PLATFORM}`);
})();
