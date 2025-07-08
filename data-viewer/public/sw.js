// Service Worker for Image Caching
const CACHE_NAME = 'alzheimer-images-v1';
const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

// URLs to cache on install
const urlsToCache = [
  '/',
  '/index.html',
  '/src/main.tsx'
];

// Install event - cache essential resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('Service Worker: Opened cache');
        return cache.addAll(urlsToCache);
      })
      .then(() => {
        console.log('Service Worker: Essential resources cached');
        return self.skipWaiting();
      })
  );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('Service Worker: Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('Service Worker: Activated');
      return self.clients.claim();
    })
  );
});

// Fetch event - implement caching strategy for images
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Only handle GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Handle CloudFront image requests with cache-first strategy
  if (url.hostname.includes('cloudfront.net') || url.pathname.includes('/assets/images/')) {
    event.respondWith(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.match(request).then((cachedResponse) => {
          // Check if we have a cached version
          if (cachedResponse) {
            // Check if cache is still valid (not expired)
            const cachedDate = new Date(cachedResponse.headers.get('sw-cached-date') || 0);
            const now = new Date();
            
            if (now.getTime() - cachedDate.getTime() < CACHE_EXPIRY) {
              console.log('Service Worker: Serving from cache:', request.url);
              return cachedResponse;
            } else {
              console.log('Service Worker: Cache expired, fetching fresh:', request.url);
              // Cache expired, remove it
              cache.delete(request);
            }
          }

          // Fetch from network and cache the response
          return fetch(request).then((networkResponse) => {
            // Only cache successful responses
            if (networkResponse.status === 200) {
              // Clone the response before caching (response can only be consumed once)
              const responseToCache = networkResponse.clone();
              
              // Add timestamp header for cache expiry
              const headers = new Headers(responseToCache.headers);
              headers.set('sw-cached-date', new Date().toISOString());
              
              const modifiedResponse = new Response(responseToCache.body, {
                status: responseToCache.status,
                statusText: responseToCache.statusText,
                headers: headers
              });

              // Cache the response
              cache.put(request, modifiedResponse.clone()).then(() => {
                console.log('Service Worker: Cached:', request.url);
              });
              
              return networkResponse;
            }
            
            return networkResponse;
          }).catch((error) => {
            console.log('Service Worker: Fetch failed:', error);
            // Return cached version if network fails
            return cachedResponse || new Response('Network error', { status: 503 });
          });
        });
      })
    );
    return;
  }

  // For non-image requests, use network-first strategy
  event.respondWith(
    fetch(request).catch(() => {
      return caches.match(request);
    })
  );
});

// Message handler for manual cache management
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.delete(CACHE_NAME).then(() => {
        console.log('Service Worker: Cache cleared');
        event.ports[0].postMessage({ success: true });
      })
    );
  }
});