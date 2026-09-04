export class AnalysisCache {
    constructor(limit = 100) {
        this.cache = new Map();
        this.limit = limit;
    }
    get(key) { return this.cache.get(key) || null; }
    set(key, data) { this.cache.set(key, data); }
    has(key) { return this.cache.has(key); }
    clear() { this.cache.clear(); }
}