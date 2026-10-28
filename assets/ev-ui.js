/* EviCortex site — interaction layer. Works without GSAP/Lenis (content stays visible). */
(function () {
  'use strict';
  var d = document, html = d.documentElement, body = d.body;
  var $ = function (s, c) { return (c || d).querySelector(s); };
  var $$ = function (s, c) { return Array.prototype.slice.call((c || d).querySelectorAll(s)); };
  var reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  var fine = matchMedia('(pointer: fine)').matches;
  var G = window.gsap, ST = window.ScrollTrigger, anim = !!(G && ST) && !reduce;
  var root = body.dataset.root || '';
  var EV = window.EV = window.EV || {};
  function scene(fn) { if (EV.sceneReady) fn(); else d.addEventListener('ev:scene', fn, { once: true }); }

  /* ---------- arrival veil ---------- */
  var veil = $('.veil');
  var arrived = false; try { arrived = sessionStorage.getItem('ev-zoom') === '1'; } catch (e) {}
  if (veil && arrived) { veil.classList.add('arrive'); requestAnimationFrame(function () { requestAnimationFrame(function () { veil.classList.add('out'); }); }); setTimeout(function () { veil.className = 'veil'; }, 1500); }
  addEventListener('pageshow', function (e) { if (e.persisted && veil) veil.className = 'veil'; });

  /* ---------- smooth scroll ---------- */
  var lenis = null;
  if (window.Lenis && !reduce) {
    lenis = new Lenis({ duration: 1.15, easing: function (t) { return Math.min(1, 1.001 - Math.pow(2, -10 * t)); }, smoothWheel: true });
    if (anim) { lenis.on('scroll', ST.update); G.ticker.add(function (t) { lenis.raf(t * 1000); }); G.ticker.lagSmoothing(0); }
    else { (function raf(t) { lenis.raf(t); requestAnimationFrame(raf); })(0); }
  }
  if (anim) G.registerPlugin(ST);
  function scrollTo(target) {
    if (typeof target === 'number') { if (lenis) lenis.scrollTo(target, { duration: 1.4 }); else window.scrollTo({ top: target, behavior: reduce ? 'auto' : 'smooth' }); return; }
    var el = typeof target === 'string' ? d.getElementById(target.replace(/^#/, '')) : target;
    if (!el) return;
    if (lenis) lenis.scrollTo(el, { offset: -80, duration: 1.6 }); else el.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth' });
  }

  /* ---------- header: progress, hide on scroll, mobile menu ---------- */
  var hdr = $('.hdr'), prog = $('.hdr .prog'), lastY = 0;
  function onScroll() {
    var y = scrollY, h = d.documentElement.scrollHeight - innerHeight;
    if (prog) prog.style.transform = 'scaleX(' + (h > 0 ? y / h : 0) + ')';
    if (hdr) hdr.classList.toggle('hide', y > lastY && y > 300 && !$('.nav.open'));
    lastY = y;
  }
  addEventListener('scroll', onScroll, { passive: true }); onScroll();
  var menu = $('.menu-btn'), nav = $('.nav');
  if (menu) menu.addEventListener('click', function () { var o = nav.classList.toggle('open'); menu.setAttribute('aria-expanded', o); });

  /* ---------- page transitions: zoom through a node ---------- */
  d.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[href]'); if (!a) return;
    var href = a.getAttribute('href');
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey || a.target === '_blank' || a.hasAttribute('download')) return;
    if (href.charAt(0) === '#') { e.preventDefault(); scrollTo(href); if (nav) nav.classList.remove('open'); history.replaceState(null, '', href); return; }
    var url; try { url = new URL(a.href, location.href); } catch (err) { return; }
    if (url.origin !== location.origin || /\.(json|md|py|txt|pdf)$/i.test(url.pathname)) return;
    if (url.pathname === location.pathname && url.hash) { e.preventDefault(); scrollTo(url.hash); return; }
    e.preventDefault();
    try { sessionStorage.setItem('ev-zoom', '1'); } catch (err) {}
    if (veil) { veil.className = 'veil'; void veil.offsetWidth; setTimeout(function () { veil.classList.add('on'); }, reduce ? 0 : 380); }
    var go = function () { location.href = url.href; };
    if (EV.dive && !reduce) EV.dive(950).then(go); else setTimeout(go, reduce ? 150 : 900);
  });

  /* ---------- custom cursor + magnetic ---------- */
  if (fine && !reduce) {
    html.classList.add('has-cursor');
    var dot = d.createElement('div'), ring = d.createElement('div');
    dot.className = 'cur cur-dot'; ring.className = 'cur cur-ring'; body.append(dot, ring);
    var mx = -100, my = -100, rx = -100, ry = -100;
    addEventListener('pointermove', function (e) { mx = e.clientX; my = e.clientY; dot.style.transform = 'translate3d(' + mx + 'px,' + my + 'px,0)'; }, { passive: true });
    (function loop() { rx += (mx - rx) * .18; ry += (my - ry) * .18; ring.style.transform = 'translate3d(' + rx + 'px,' + ry + 'px,0)'; requestAnimationFrame(loop); })();
    d.addEventListener('pointerover', function (e) { ring.classList.toggle('hover', !!e.target.closest('a,button,[data-magnetic],summary,input')); });
    d.addEventListener('pointerdown', function () { ring.classList.add('down'); });
    d.addEventListener('pointerup', function () { ring.classList.remove('down'); });
    $$('[data-magnetic]').forEach(function (el) {
      el.addEventListener('pointermove', function (e) {
        var r = el.getBoundingClientRect(), x = e.clientX - r.left - r.width / 2, y = e.clientY - r.top - r.height / 2;
        el.style.transform = 'translate(' + x * .28 + 'px,' + y * .38 + 'px)';
        el.style.setProperty('--mx', (e.clientX - r.left) + 'px'); el.style.setProperty('--my', (e.clientY - r.top) + 'px');
      });
      el.addEventListener('pointerleave', function () { el.style.transition = 'transform .6s cubic-bezier(.2,.8,.2,1)'; el.style.transform = ''; setTimeout(function () { el.style.transition = ''; }, 600); });
    });
    $$('[data-tilt]').forEach(function (el) {
      el.addEventListener('pointermove', function (e) {
        var r = el.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - .5, y = (e.clientY - r.top) / r.height - .5;
        el.style.transform = 'rotateY(' + x * 10 + 'deg) rotateX(' + (-y * 10) + 'deg) translateZ(0)';
        el.style.setProperty('--mx', (x + .5) * 100 + '%'); el.style.setProperty('--my', (y + .5) * 100 + '%');
      });
      el.addEventListener('pointerleave', function () { el.style.transform = ''; });
    });
  }

  /* ---------- kinetic typography ---------- */
  function split(el) {
    if (el.dataset.done) return $$('.ch', el); el.dataset.done = 1;
    var walk = function (node) {
      Array.prototype.slice.call(node.childNodes).forEach(function (n) {
        if (n.nodeType === 3) {
          var frag = d.createDocumentFragment();
          n.textContent.split(/(\s+)/).forEach(function (w) {
            if (!w) return;
            if (/^\s+$/.test(w)) { frag.appendChild(d.createTextNode(' ')); return; }
            var ws = d.createElement('span'); ws.className = 'w';
            Array.prototype.forEach.call(w, function (c) { var s = d.createElement('span'); s.className = 'ch'; s.textContent = c; ws.appendChild(s); });
            frag.appendChild(ws);
          });
          n.replaceWith(frag);
        } else if (n.nodeType === 1 && !n.classList.contains('w')) walk(n);
      });
    };
    var label = el.textContent.replace(/\s+/g, ' ').trim();
    walk(el); el.classList.add('split-mask');
    $$('.w', el).forEach(function (w) { w.setAttribute('aria-hidden', 'true'); });
    var sr = d.createElement('span'); sr.className = 'sr'; sr.textContent = label; el.appendChild(sr);
    return $$('.ch', el);
  }
  function scramble(el) {
    var txt = el.dataset.text || el.textContent, chars = '01#*/<>_{}=+ABCDEFXYZ', f = 0, total = 22;
    el.dataset.text = txt;
    var iv = setInterval(function () {
      f++; var out = '';
      for (var i = 0; i < txt.length; i++) out += (i < txt.length * f / total || txt[i] === ' ') ? txt[i] : chars[Math.floor(Math.random() * chars.length)];
      el.textContent = out; if (f >= total) { clearInterval(iv); el.textContent = txt; }
    }, 34);
  }
  if (anim) {
    $$('[data-split]').forEach(function (el) {
      var chars = split(el), hero = el.hasAttribute('data-hero');
      G.set(chars, { yPercent: 115, rotate: 8, opacity: 0 });
      var tw = { yPercent: 0, rotate: 0, opacity: 1, duration: 1.1, ease: 'expo.out', stagger: { each: hero ? .022 : .012 }, delay: hero ? (arrived ? .5 : .25) : 0 };
      if (hero) G.to(chars, tw); else G.to(chars, Object.assign(tw, { scrollTrigger: { trigger: el, start: 'top 85%' } }));
    });
    $$('[data-scramble]').forEach(function (el) { ST.create({ trigger: el, start: 'top 90%', once: true, onEnter: function () { scramble(el); } }); });
    $$('[data-reveal]').forEach(function (el) {
      var kids = el.hasAttribute('data-stagger') ? Array.prototype.slice.call(el.children) : [el];
      G.fromTo(kids, { y: 46, opacity: 0 }, { y: 0, opacity: 1, duration: 1.1, ease: 'power3.out', stagger: .09, scrollTrigger: { trigger: el, start: 'top 86%' } });
    });
  }

  /* ---------- count-ups ---------- */
  $$('[data-count]').forEach(function (el) {
    var to = parseFloat(el.dataset.count), from = parseFloat(el.dataset.from || 0), dec = +(el.dataset.dec || 0), out = $('.cv', el);
    if (!anim) { out.textContent = to.toFixed(dec); return; }
    var o = { v: from }; out.textContent = from.toFixed(dec);
    G.to(o, { v: to, duration: 2.2, ease: 'power3.out', scrollTrigger: { trigger: el, start: 'top 85%' }, onUpdate: function () { out.textContent = o.v.toFixed(dec); } });
  });

  /* ---------- charts ---------- */
  if (anim) {
    $$('.ln-path').forEach(function (p) {
      var L = p.getTotalLength(); var dashed = p.classList.contains('ln-base');
      if (dashed) { G.from(p, { opacity: 0, duration: 1.2, scrollTrigger: { trigger: p.closest('svg'), start: 'top 80%' } }); return; }
      G.fromTo(p, { strokeDasharray: L, strokeDashoffset: L }, { strokeDashoffset: 0, duration: 2, ease: 'power2.inOut', scrollTrigger: { trigger: p.closest('svg'), start: 'top 80%' } });
    });
    $$('.chart svg').forEach(function (s) {
      var pts = $$('.pt,.vlab text,.band,.band-lab', s); if (pts.length) G.from(pts, { opacity: 0, scale: .4, transformOrigin: '50% 50%', duration: .7, stagger: .08, delay: .6, scrollTrigger: { trigger: s, start: 'top 80%' } });
      var bars = $$('.bar', s); if (bars.length) G.from(bars, { scaleY: 0, transformOrigin: '50% 100%', duration: 1.2, ease: 'expo.out', stagger: .1, scrollTrigger: { trigger: s, start: 'top 80%' } });
    });
    $$('.hb .fl').forEach(function (f, i) { G.from(f, { scaleX: 0, duration: 1.6, ease: 'expo.out', delay: i * .15, scrollTrigger: { trigger: f, start: 'top 90%' } }); });
    $$('.road').forEach(function (r) { var ln = $('.draw-line', r); if (ln) G.to(ln, { scaleY: 1, ease: 'none', scrollTrigger: { trigger: r, start: 'top 70%', end: 'bottom 60%', scrub: true } }); });
  }

  /* ---------- hero token meter ---------- */
  var mVal = $('#meterVal'), mFill = $('#meterFill'), mNs = $('#meterNs');
  if (mVal) {
    var nss = ['assistant/user-ada', 'agent/planner', 'app/support'], ni = 0;
    var fill = function () {
      var target = 380 + Math.floor(Math.random() * 132); mNs.textContent = 'Capsule for ' + nss[ni++ % 3];
      if (!anim) { mVal.textContent = target; mFill.style.width = target / 5.12 + '%'; return; }
      var o = { v: 0 }; G.to(o, { v: target, duration: 1.8, ease: 'power3.out', onUpdate: function () { mVal.textContent = Math.round(o.v); mFill.style.width = o.v / 5.12 + '%'; } });
      scene(function () { EV.pulse && EV.pulse(10); });
    };
    fill(); if (!reduce) setInterval(fill, 4200);
  }

  /* ---------- scene modes per section ---------- */
  if (anim) {
    $$('[data-mode]').forEach(function (sec) {
      var mode = sec.dataset.mode;
      ST.create({
        trigger: sec, start: 'top 55%', end: 'bottom 45%',
        onToggle: function (s) { if (s.isActive) scene(function () { EV.setMode(mode, s.progress); }); },
        onUpdate: function (s) { if (EV.setProgress && s.isActive) EV.setProgress(s.progress); }
      });
    });
  }

  /* ---------- story ---------- */
  var story = $('.story');
  if (story) {
    var steps = $$('.step', story), dots = $$('.story-dots b', story), cur = -1;
    var setStep = function (p) {
      var n = steps.length, i = Math.min(n - 1, Math.floor(p * n));
      dots.forEach(function (b, k) { b.style.transform = 'scaleX(' + Math.max(0, Math.min(1, p * n - k)) + ')'; });
      if (i !== cur) { cur = i; steps.forEach(function (s, k) { s.classList.toggle('on', k === i); }); }
      scene(function () { EV.setStory(Math.max(0, p * n - 1), Math.max(0, Math.min(1, (p * n - 2) * 1.4))); });
    };
    if (anim) ST.create({ trigger: story, start: 'top top', end: 'bottom bottom', onUpdate: function (s) { setStep(s.progress); }, onEnter: function () { scene(function () { EV.setMode('story'); }); }, onEnterBack: function () { scene(function () { EV.setMode('story'); }); } });
    else steps.forEach(function (s) { s.classList.add('on'); });
    if (anim) setStep(0);
  }

  /* ---------- pipeline ---------- */
  var pipe = $('.pipe-sec');
  if (pipe) {
    var data = JSON.parse($('#pipe-data').textContent), btns = $$('.rail button', pipe), si = -1;
    var num = $('.stage-num', pipe), nm = $('.stage-name', pipe), api = $('.stage-api', pipe), bd = $('.stage-body', pipe), card = $('.stage-card', pipe);
    var show = function (i) {
      if (i === si) return; si = i; var s = data[i];
      btns.forEach(function (b, k) { b.classList.toggle('on', k === i); b.setAttribute('aria-current', k === i ? 'step' : 'false'); });
      var swap = function () { num.textContent = '0' + (i + 1); nm.textContent = s[0]; api.textContent = s[1]; bd.innerHTML = s[2]; };
      if (anim) G.timeline().to(card, { opacity: 0, y: 14, duration: .2, ease: 'power2.in' }).add(swap).to(card, { opacity: 1, y: 0, duration: .45, ease: 'power3.out' });
      else swap();
    };
    show(0);
    var pst = anim ? ST.create({
      trigger: pipe, start: 'top top', end: 'bottom bottom',
      onUpdate: function (s) {
        var p = s.progress; show(Math.min(6, Math.floor(p * 7)));
        $$('.rail b', pipe).forEach(function (b, k) { b.style.transform = 'scaleX(' + Math.max(0, Math.min(1, p * 7 - k)) + ')'; });
        scene(function () { EV.setMode('pipeline', p); });
      },
      onLeave: function () { scene(function () { EV.setMode('overview', 0); }); },
      onLeaveBack: function () { scene(function () { EV.setMode('story', 1); }); }
    }) : null;
    btns.forEach(function (b, k) {
      b.addEventListener('click', function () {
        if (pst) { var y = pst.start + (pst.end - pst.start) * ((k + .5) / 7); scrollTo(y); }
        else show(k);
      });
    });
  }

  /* ---------- tabs & copy ---------- */
  $$('.term').forEach(function (t) {
    var tabs = $$('[role=tab]', t);
    tabs.forEach(function (b) {
      b.addEventListener('click', function () {
        tabs.forEach(function (x) { var on = x === b; x.setAttribute('aria-selected', on); x.tabIndex = on ? 0 : -1; d.getElementById(x.getAttribute('aria-controls')).hidden = !on; });
      });
    });
  });
  function copyText(btn, text) {
    var done = function () { var l = btn.textContent; btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = l; }, 1600); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, done); else done();
  }
  $$('.term .copy').forEach(function (b) {
    b.addEventListener('click', function () { var p = $('.tabpanel:not([hidden]) pre', b.closest('.term')); copyText(b, p.innerText); });
  });
  $$('.prose pre').forEach(function (pre) {
    var w = pre.closest('.codewrap'); if (!w) return;
    var b = d.createElement('button'); b.className = 'copy'; b.type = 'button'; b.textContent = 'Copy';
    b.addEventListener('click', function () { copyText(b, pre.innerText); }); w.appendChild(b);
  });

  /* ---------- docs: drawer, toc scrollspy, search ---------- */
  var side = $('.side'), sb = $('.side-btn');
  if (sb && side) sb.addEventListener('click', function () { side.classList.toggle('open'); });
  var tocLinks = $$('.toc a');
  if (tocLinks.length) {
    var heads = tocLinks.map(function (a) { return d.getElementById(decodeURIComponent(a.hash.slice(1))); });
    var spy = function () {
      var y = 120, idx = 0;
      heads.forEach(function (h, i) { if (h && h.getBoundingClientRect().top < y) idx = i; });
      tocLinks.forEach(function (a, i) { a.classList.toggle('on', i === idx); });
    };
    addEventListener('scroll', spy, { passive: true }); spy();
  }
  if (anim && $('.prose')) {
    $$('.prose > :not(h1)').forEach(function (el) {
      G.from(el, { y: 24, opacity: 0, duration: .8, ease: 'power3.out', scrollTrigger: { trigger: el, start: 'top 92%', once: true } });
    });
  }

  var sBox = $('.search'), sIn = sBox && $('input', sBox), sList = sBox && $('ul', sBox), sel = 0, hits = [];
  function openSearch() { if (!sBox) return; sBox.classList.add('open'); sIn.value = ''; render(''); setTimeout(function () { sIn.focus(); }, 30); if (lenis) lenis.stop(); }
  function closeSearch() { sBox.classList.remove('open'); if (lenis) lenis.start(); }
  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function render(q) {
    var idx = window.EV_SEARCH || [], terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) hits = idx.filter(function (e) { return !e.a; }).slice(0, 14);
    else hits = idx.map(function (e) {
      var hay = (e.t + ' ' + e.h + ' ' + e.x).toLowerCase(), sc = 0;
      for (var i = 0; i < terms.length; i++) { if (hay.indexOf(terms[i]) < 0) return null; sc += (e.h.toLowerCase().indexOf(terms[i]) >= 0 ? 3 : 0) + (e.t.toLowerCase().indexOf(terms[i]) >= 0 ? 2 : 0) + 1; }
      return { e: e, sc: sc };
    }).filter(Boolean).sort(function (a, b) { return b.sc - a.sc; }).slice(0, 20).map(function (h) { return h.e; });
    sel = 0;
    var mark = function (s) { s = esc(s); terms.forEach(function (t) { s = s.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>'); }); return s; };
    sList.innerHTML = hits.length ? hits.map(function (e, i) {
      return '<li><a href="' + root + e.u + '"' + (i === 0 ? ' class="sel"' : '') + '><small>' + esc(e.t) + '</small>' + mark(e.h) + (e.x ? '<span>' + mark(e.x) + '</span>' : '') + '</a></li>';
    }).join('') : '<li class="empty">No matches. Try another word.</li>';
  }
  if (sBox) {
    $$('.search-btn').forEach(function (b) { b.addEventListener('click', openSearch); });
    sBox.addEventListener('click', function (e) { if (e.target === sBox) closeSearch(); });
    sIn.addEventListener('input', function () { render(sIn.value); });
    sIn.addEventListener('keydown', function (e) {
      var as = $$('a', sList);
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); sel = (sel + (e.key === 'ArrowDown' ? 1 : -1) + as.length) % as.length; as.forEach(function (a, i) { a.classList.toggle('sel', i === sel); }); if (as[sel]) as[sel].scrollIntoView({ block: 'nearest' }); }
      if (e.key === 'Enter' && as[sel]) { e.preventDefault(); closeSearch(); as[sel].click(); }
    });
    d.addEventListener('keydown', function (e) {
      if ((e.key === 'k' && (e.metaKey || e.ctrlKey)) || (e.key === '/' && !/INPUT|TEXTAREA/.test(d.activeElement.tagName))) { e.preventDefault(); openSearch(); }
      if (e.key === 'Escape' && sBox.classList.contains('open')) closeSearch();
    });
  }

  /* ---------- in-page hash on load ---------- */
  if (location.hash) setTimeout(function () { scrollTo(location.hash); }, 400);
  if (anim) addEventListener('load', function () { ST.refresh(); });
})();
