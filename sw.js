const CACHE_NAME = 'utt-ai-safety-v1';
const ASSETS_TO_CACHE = [
  '/',
  '/index.html',
  '/manifest.json',
  '/css/style.css',
  '/src/script.js',
  '/assets/logo.png',
  '/assets/warning.WAV',
  '/assets/alarm.WAV',
  '/assets/video7.mp4',
  '/models/best.onnx',
  '/models/cone_sign.onnx',
  '/models/tuthenga.onnx',
  'https://unpkg.com/lucide@latest',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js',
  'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Opened cache');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      if (response) {
        return response; // Trả về từ cache
      }
      return fetch(event.request).then(
        (response) => {
          // Chỉ cache lại model ONNX nếu được tải lần đầu
          if(!response || response.status !== 200 || response.type !== 'basic') {
            if(event.request.url.includes('.onnx')) {
                let responseToCache = response.clone();
                caches.open(CACHE_NAME)
                  .then((cache) => {
                    cache.put(event.request, responseToCache);
                  });
            }
            return response;
          }
          return response;
        }
      );
    })
  );
});

self.addEventListener('activate', (event) => {
  const cacheWhitelist = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheWhitelist.indexOf(cacheName) === -1) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
});
