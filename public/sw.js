/* Ghana Health AI service worker — production-only offline shell. */
const CACHE = "gha-shell-v3";
const SHELL = ["/", "/chat", "/voice", "/login", "/manifest.webmanifest"];

function isLocalDev() {
  return self.location.hostname === "localhost" || self.location.hostname === "127.0.0.1";
}

self.addEventListener("install", (event) => {
  if (isLocalDev()) {
    event.waitUntil(self.skipWaiting());
    return;
  }

  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE || isLocalDev())
            .map((key) => caches.delete(key)),
        ),
      )
      .then(() => (isLocalDev() ? self.registration.unregister() : self.clients.claim())),
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (
    isLocalDev() ||
    url.origin !== self.location.origin ||
    url.pathname.startsWith("/api/") ||
    url.pathname.startsWith("/_next/")
  ) {
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((res) => {
          const clone = res.clone();
          caches.open(CACHE).then((cache) => cache.put(request, clone));
          return res;
        })
        .catch(() => caches.match(request).then((cached) => cached || caches.match("/"))),
    );
    return;
  }

  event.respondWith(
    fetch(request)
      .then((res) => {
        if (res.ok) {
          const clone = res.clone();
          caches.open(CACHE).then((cache) => cache.put(request, clone));
        }
        return res;
      })
      .catch(() => caches.match(request)),
  );
});
