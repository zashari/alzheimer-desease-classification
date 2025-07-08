// Service Worker for Image Caching
const CACHE_NAME = 'alzheimer-images-v1';
const CACHE_EXPIRY = 24 * 60 * 60 * 1000; // 24 hours in milliseconds

// Progress tracking
let imageCacheProgress = {
  total: 0,
  cached: 0,
  currentBatch: new Set()
};

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
                
                // Update progress tracking
                imageCacheProgress.cached++;
                broadcastProgress();
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

// Function to broadcast progress to all clients
function broadcastProgress() {
  const message = {
    type: 'IMAGE_CACHE_PROGRESS',
    cached: imageCacheProgress.cached,
    total: imageCacheProgress.total
  };
  
  // Send to all clients
  self.clients.matchAll().then(clients => {
    clients.forEach(client => {
      client.postMessage(message);
    });
  });
}

// Message handler for manual cache management and progress tracking
self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'CLEAR_CACHE') {
    event.waitUntil(
      caches.delete(CACHE_NAME).then(() => {
        console.log('Service Worker: Cache cleared');
        event.ports[0].postMessage({ success: true });
      })
    );
  } else if (event.data && event.data.type === 'START_TRACKING') {
    // Initialize progress tracking for a new batch
    imageCacheProgress.total = event.data.total || 0;
    imageCacheProgress.cached = 0;
    imageCacheProgress.currentBatch.clear();
    console.log('Service Worker: Started tracking', imageCacheProgress.total, 'images');
  } else if (event.data && event.data.type === 'BATCH_PROGRESS') {
    // Forward batch progress to all clients
    const message = {
      type: 'BATCH_PROGRESS',
      batched: event.data.batched,
      total: event.data.total
    };
    
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage(message);
      });
    });
  } else if (event.data && event.data.type === 'LOADING_COMPLETE') {
    // Forward loading complete to all clients
    const message = {
      type: 'LOADING_COMPLETE'
    };
    
    self.clients.matchAll().then(clients => {
      clients.forEach(client => {
        client.postMessage(message);
      });
    });
  }
});