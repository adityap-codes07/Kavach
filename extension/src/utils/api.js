export class KavachAPI {
    constructor(baseUrl = "http://localhost:8000/api/v1") {
        this.baseUrl = baseUrl;
    }

    async analyzeText(payload) {
        let response;
        try {
            response = await fetch(`${this.baseUrl}/analyze/text`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
        } catch (err) {
            response = await fetch(`${this.baseUrl}/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
        }

        if (!response.ok && response.status === 404) {
            const fallback = await fetch(`${this.baseUrl}/analyze`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (fallback.ok) return await fallback.json();
        }

        if (!response.ok) throw new Error(`API Error: ${response.status}`);
        return await response.json();
    }

    async analyzeFile(fileData, filename) {
        const formData = new FormData();
        formData.append("file", fileData, filename);

        const response = await fetch(`${this.baseUrl}/analyze/file`, {
            method: "POST",
            body: formData,
        });
        if (!response.ok) throw new Error(`File API Error: ${response.status}`);
        return await response.json();
    }

    async healthCheck() {
        const rootUrl = this.baseUrl.replace(/\/api\/v1\/?$/, "");
        const endpoints = [
            `${this.baseUrl}/health`,
            `${rootUrl}/health`,
            `${rootUrl}/docs`,
        ];

        for (const endpoint of endpoints) {
            try {
                const response = await fetch(endpoint, { method: "GET" });
                if (response.ok) {
                    return { status: "healthy" };
                }
            } catch {
                // Try next endpoint
            }
        }
        return { status: "offline" };
    }

    async getBenchmark() {
        try {
            const response = await fetch(`${this.baseUrl}/models/benchmark`);
            if (response.ok) return await response.json();
        } catch {}

        try {
            const response = await fetch(`${this.baseUrl}/benchmark`);
            if (response.ok) return await response.json();
        } catch {}

        return { latency_ms: 0, model: "bert-base-uncased (local)" };
    }
}