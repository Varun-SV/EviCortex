// EviCortex cinematic background: a namespaced claim graph, a pipeline corridor and a camera director.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const canvas = document.getElementById('ev-canvas');
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const AMBER = new THREE.Color('#f6b84a'), ROSE = new THREE.Color('#ea6f95'), LILAC = new THREE.Color('#a9b1ff');

let renderer;
try { renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' }); }
catch (e) { document.documentElement.classList.add('no-webgl'); throw e; }
const lite = /[?&]fx=lite/.test(location.search);
renderer.setPixelRatio(lite ? .5 : Math.min(devicePixelRatio, 2));
renderer.setClearColor(0x000000, 1); // composer output re-encodes; true black lands as black

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x000000, 0.012);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 400);
camera.position.set(0, 6, 60);

// Seeded random so every page shows the same constellation.
let seed = 7; const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const gauss = () => (rnd() + rnd() + rnd() - 1.5) / 1.5;

/* ---------- stars ---------- */
{
  const n = 3200, p = new Float32Array(n * 3), s = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const r = 70 + rnd() * 160, th = rnd() * Math.PI * 2, ph = Math.acos(2 * rnd() - 1);
    p.set([r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th) * 0.6, r * Math.cos(ph) - 40], i * 3);
    s[i] = 0.3 + rnd() * 1.1;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(p, 3));
  g.setAttribute('aSize', new THREE.BufferAttribute(s, 1));
  const m = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uPR: { value: renderer.getPixelRatio() } },
    vertexShader: `attribute float aSize; uniform float uTime; uniform float uPR; varying float vA;
      void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv;
      vA = .45+.55*sin(uTime*.8+position.x*1.7+position.y);
      gl_PointSize = min(aSize*uPR*(120./-mv.z), 6.*uPR); }`,
    fragmentShader: `varying float vA; void main(){ float d=length(gl_PointCoord-.5); if(d>.5) discard;
      gl_FragColor = vec4(vec3(.78,.8,1.), pow(smoothstep(.5,0.,d),2.)*vA*.55); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, fog: false
  });
  scene.add(new THREE.Points(g, m)); var starMat = m;
}

/* ---------- claim graph: three namespaces ---------- */
const CL = [
  { c: new THREE.Vector3(-13, 4, -8), r: 5.6, n: 120 },
  { c: new THREE.Vector3(9, -1, -2), r: 6.4, n: 160 },   // the "story" namespace
  { c: new THREE.Vector3(-3, -9, -16), r: 5.4, n: 110 }
];
const nodes = [];
CL.forEach((cl, ci) => {
  for (let i = 0; i < cl.n; i++) {
    const v = new THREE.Vector3(gauss(), gauss() * 0.8, gauss()).multiplyScalar(cl.r).add(cl.c);
    nodes.push({ p: v, ci, stale: rnd() < 0.22 ? 1 : 0, size: 0.6 + rnd() * 1.2, phase: rnd() * 6.28, lit: 0, nb: [] });
  }
});
// Story pair: one superseded claim and the claim that replaced it, near the centre of namespace 1.
const S0 = CL[1].c;
const storyOld = { p: S0.clone().add(new THREE.Vector3(-1.6, 0.6, 1.5)), ci: 1, stale: 0, size: 2.6, phase: 0, lit: 0, nb: [], story: 1 };
const storyNew = { p: S0.clone().add(new THREE.Vector3(1.7, -0.3, 1.8)), ci: 1, stale: 0, size: 2.6, phase: 1, lit: 0, nb: [], story: 2 };
nodes.push(storyOld, storyNew);

// Edges: link each node to its two nearest neighbours inside its own namespace (no cross-namespace edges).
const edges = [];
nodes.forEach((a, i) => {
  const d = [];
  nodes.forEach((b, j) => { if (i !== j && a.ci === b.ci) d.push([a.p.distanceToSquared(b.p), j]); });
  d.sort((x, y) => x[0] - y[0]);
  for (let k = 0; k < 2; k++) { const j = d[k][1]; if (!a.nb.includes(j)) { a.nb.push(j); nodes[j].nb.push(i); edges.push([i, j]); } }
});

const N = nodes.length;
const nPos = new Float32Array(N * 3), nSize = new Float32Array(N), nPhase = new Float32Array(N), nStale = new Float32Array(N), nLit = new Float32Array(N), nStory = new Float32Array(N);
nodes.forEach((o, i) => { nPos.set([o.p.x, o.p.y, o.p.z], i * 3); nSize[i] = o.size; nPhase[i] = o.phase; nStale[i] = o.stale; nStory[i] = o.story || 0; });
const nGeo = new THREE.BufferGeometry();
nGeo.setAttribute('position', new THREE.BufferAttribute(nPos, 3));
nGeo.setAttribute('aSize', new THREE.BufferAttribute(nSize, 1));
nGeo.setAttribute('aPhase', new THREE.BufferAttribute(nPhase, 1));
nGeo.setAttribute('aStale', new THREE.BufferAttribute(nStale, 1));
nGeo.setAttribute('aStory', new THREE.BufferAttribute(nStory, 1));
const litAttr = new THREE.BufferAttribute(nLit, 1); litAttr.setUsage(THREE.DynamicDrawUsage);
nGeo.setAttribute('aLit', litAttr);

const U = {
  uTime: { value: 0 }, uPR: { value: renderer.getPixelRatio() }, uMouse: { value: new THREE.Vector2(9, 9) },
  uAmber: { value: AMBER }, uRose: { value: ROSE }, uLilac: { value: LILAC },
  uStaleFade: { value: 0 },  // 0: stale claims visible as rose rings, 1: dissolved
  uStory: { value: 0 },      // 0..3 story step
  uDim: { value: 1 }
};
const nMat = new THREE.ShaderMaterial({
  uniforms: U, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute float aSize, aPhase, aStale, aLit, aStory;
    uniform float uTime, uPR, uStaleFade, uStory; uniform vec2 uMouse;
    varying float vStale, vA, vLit, vStory;
    void main(){
      vec3 p = position + vec3(sin(uTime*.4+aPhase), cos(uTime*.33+aPhase*1.3), sin(uTime*.27+aPhase*.7))*.18;
      vec4 mv = modelViewMatrix*vec4(p,1.); gl_Position = projectionMatrix*mv;
      vec2 sp = gl_Position.xy/gl_Position.w; float hov = smoothstep(.22,0.,distance(sp,uMouse));
      float st = aStale;
      // story node 1 becomes superseded at step >= 2
      if (aStory > .5 && aStory < 1.5) st = smoothstep(1.2,2.,uStory);
      vStale = st; vStory = aStory; vLit = aLit + hov;
      float fade = mix(1., 1.-uStaleFade, st);
      vA = (.75+.25*sin(uTime*1.3+aPhase)) * fade;
      float storyBoost = aStory > .5 ? 1. + smoothstep(.2,1.,uStory)*1.4 : 1.;
      gl_PointSize = min(aSize*storyBoost*(1.+hov*1.6+aLit*.9)*uPR*(260./-mv.z), 64.*uPR);
      vA *= smoothstep(1.5, 4., -mv.z);
    }`,
  fragmentShader: `uniform vec3 uAmber, uRose, uLilac; uniform float uDim, uStory; varying float vStale, vA, vLit, vStory;
    void main(){
      float d = length(gl_PointCoord-.5); if (d>.5) discard;
      float core = smoothstep(.5,.0,d); core = pow(core, 2.2);
      float ring = smoothstep(.5,.42,d)*smoothstep(.26,.34,d);
      vec3 cur = mix(uAmber, vec3(1.,.93,.75), vLit*.6);
      vec3 col = mix(cur, uRose, vStale);
      float a = mix(core*1.2 + vLit*core, ring*.95 + core*.15, vStale);
      // emphasise the replacing claim in the story
      if (vStory > 1.5) a *= 1. + smoothstep(1.,2.,uStory)*1.2;
      gl_FragColor = vec4(col*(1.+vLit*.8), a*vA*uDim);
    }`
});
const nodePoints = new THREE.Points(nGeo, nMat); scene.add(nodePoints);

// Edge lines with per-vertex brightness driven by node "lit".
const ePos = new Float32Array(edges.length * 6), eCol = new Float32Array(edges.length * 6);
edges.forEach(([a, b], k) => { const A = nodes[a].p, B = nodes[b].p; ePos.set([A.x, A.y, A.z, B.x, B.y, B.z], k * 6); });
const eGeo = new THREE.BufferGeometry();
eGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
const eColAttr = new THREE.BufferAttribute(eCol, 3); eColAttr.setUsage(THREE.DynamicDrawUsage);
eGeo.setAttribute('color', eColAttr);
const eMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: .55, blending: THREE.AdditiveBlending, depthWrite: false });
scene.add(new THREE.LineSegments(eGeo, eMat));

// Namespace boundaries: tilted rings.
const rings = CL.map((cl, i) => {
  const g = new THREE.BufferGeometry().setFromPoints(Array.from({ length: 129 }, (_, k) => {
    const t = k / 128 * Math.PI * 2; return new THREE.Vector3(Math.cos(t), 0, Math.sin(t)).multiplyScalar(cl.r * 1.75);
  }));
  const m = new THREE.LineDashedMaterial({ color: LILAC, dashSize: .5, gapSize: .45, transparent: true, opacity: .28, blending: THREE.AdditiveBlending, depthWrite: false });
  const l = new THREE.Line(g, m); l.computeLineDistances(); l.position.copy(cl.c); l.rotation.set(.35 + i * .2, 0, .25 - i * .15); scene.add(l); return l;
});

// Query packets that travel along edges and light the nodes they reach.
const PK = 16, pkPos = new Float32Array(PK * 3), packets = [];
for (let i = 0; i < PK; i++) { const s = Math.floor(rnd() * N); packets.push({ a: s, b: nodes[s].nb[0], t: rnd() }); }
const pkGeo = new THREE.BufferGeometry(); pkGeo.setAttribute('position', new THREE.BufferAttribute(pkPos, 3));
const pkMat = new THREE.ShaderMaterial({ uniforms: { uPR: U.uPR }, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `uniform float uPR; void main(){ vec4 mv = modelViewMatrix*vec4(position,1.); gl_Position = projectionMatrix*mv; gl_PointSize = min(2.4*uPR*(160./-mv.z), 40.*uPR); }`,
  fragmentShader: `void main(){ float d=length(gl_PointCoord-.5); if(d>.5) discard; gl_FragColor = vec4(1.,.95,.82, pow(smoothstep(.5,0.,d),1.6)); }` });
scene.add(new THREE.Points(pkGeo, pkMat));

/* ---------- pipeline corridor: seven gates and a stream of evidence ---------- */
const GATE_Z0 = -40, GATE_STEP = 14, GATES = 7, GATE_X = 34;
const gates = [];
for (let i = 0; i < GATES; i++) {
  const m = new THREE.MeshBasicMaterial({ color: LILAC, transparent: true, opacity: .35, blending: THREE.AdditiveBlending, depthWrite: false });
  const t = new THREE.Mesh(new THREE.TorusGeometry(4.6, .06, 6, 96), m);
  t.position.set(GATE_X, 0, GATE_Z0 - i * GATE_STEP); scene.add(t);
  gates.push({ t, lit: 0 });
}
const ST = 900, stA = new Float32Array(ST * 4);
for (let i = 0; i < ST; i++) { const a = rnd() * 6.28, r = Math.sqrt(rnd()) * 3.4; stA.set([Math.cos(a) * r, Math.sin(a) * r, rnd(), rnd() < .3 ? 1 : 0], i * 4); }
const stGeo = new THREE.BufferGeometry();
stGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ST * 3), 3));
stGeo.setAttribute('aD', new THREE.BufferAttribute(stA, 4));
const stMat = new THREE.ShaderMaterial({
  uniforms: { uTime: U.uTime, uPR: U.uPR, uAmber: U.uAmber, uRose: U.uRose, uOn: { value: 0 },
    uZ0: { value: GATE_Z0 + 18 }, uLen: { value: GATE_STEP * 7 + 16 }, uX: { value: GATE_X },
    uFilterZ: { value: GATE_Z0 - 4 * GATE_STEP }, uPackZ: { value: GATE_Z0 - 5 * GATE_STEP } },
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: `attribute vec4 aD; uniform float uTime, uPR, uZ0, uLen, uX, uFilterZ, uPackZ; varying float vS, vA;
    void main(){
      float z = uZ0 - mod(uTime*7. + aD.z*uLen, uLen);
      float pack = mix(1., .38, smoothstep(uPackZ+2., uPackZ-3., z));
      vec3 p = vec3(uX + aD.x*pack, aD.y*pack, z);
      vS = aD.w;
      float gone = aD.w * smoothstep(uFilterZ+1.5, uFilterZ-1.5, z);
      vA = (1.-gone) * smoothstep(uZ0, uZ0-6., z) * smoothstep(uZ0-uLen, uZ0-uLen+8., z);
      vec4 mv = modelViewMatrix*vec4(p,1.); gl_Position = projectionMatrix*mv;
      gl_PointSize = min((1.4 + gone*4.)*uPR*(90./-mv.z), 26.*uPR);
      vA *= smoothstep(1., 3.5, -mv.z);
    }`,
  fragmentShader: `uniform vec3 uAmber, uRose; uniform float uOn; varying float vS, vA;
    void main(){ float d=length(gl_PointCoord-.5); if(d>.5) discard;
      gl_FragColor = vec4(mix(uAmber,uRose,vS), smoothstep(.5,0.,d)*vA*uOn); }`
});
scene.add(new THREE.Points(stGeo, stMat));

/* ---------- post-processing ---------- */
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.95, 0.6, 0.32);
composer.addPass(bloom);
composer.addPass(new OutputPass());

/* ---------- camera director ---------- */
const state = { mode: document.body.dataset.scene || 'hero', p: 0, mx: 0, my: 0, smx: 0, smy: 0 };
const desired = { pos: new THREE.Vector3(), look: new THREE.Vector3(), fov: 55 };
const look = new THREE.Vector3(0, 0, -4);
let dive = null; // {target, t0}

function direct(t) {
  const P = desired.pos, L = desired.look; let fov = 55;
  const m = state.mode, p = state.p;
  if (m === 'story') {
    const k = Math.min(1, p * 1.6);
    P.set(S0.x + 2 - k * 1.5, S0.y + 2.5 - k * 1.2, S0.z + 26 - k * 13); L.copy(S0).add(new THREE.Vector3(0, 0, 1.4)); fov = 50;
  } else if (m === 'pipeline') {
    const z = GATE_Z0 + 16 - p * (GATE_STEP * 6 + 18);
    P.set(GATE_X + Math.sin(p * 6) * .8, .6 + Math.cos(p * 5) * .4, z); L.set(GATE_X, 0, z - 14); fov = 62;
  } else if (m === 'overview') {
    const a = t * .03 + p * 1.2;
    P.set(Math.sin(a) * 34, 16 - p * 6, Math.cos(a) * 34 - 6); L.set(0, -2, -6); fov = 52;
  } else if (m === 'docs') {
    const a = t * .02;
    P.set(Math.sin(a) * 42, 10, Math.cos(a) * 42 - 6); L.set(0, -1, -6); fov = 50;
  } else { // hero
    const a = t * .045;
    const wide = innerWidth > 900; P.set(Math.sin(a) * 23 + (wide ? -6 : 0), 4 + Math.sin(t * .1) * 1.5, Math.cos(a) * 23 - 2); L.set(wide ? -15 : -2, wide ? -1 : 4, -6); fov = 55;
  }
  P.x += state.smx * 2.2; P.y += state.smy * 1.4;
  desired.fov = fov;
}

/* ---------- loop ---------- */
const clock = new THREE.Clock(); let visible = true, tAcc = 0, tScale = 1;
const tmp = new THREE.Vector3();
function frame() {
  requestAnimationFrame(frame);
  if (!visible) return;
  const dt0 = Math.min(clock.getDelta(), .05);
  const calm = state.mode === 'overview' || state.mode === 'docs';
  tScale += ((calm ? .35 : 1) * (reduce ? .25 : 1) - tScale) * (1 - Math.exp(-dt0 * 2));
  const dt = dt0 * tScale; tAcc += dt; const t = tAcc;
  U.uTime.value = t; starMat.uniforms.uTime.value = t;
  state.smx += (state.mx - state.smx) * (1 - Math.exp(-dt * 3)); state.smy += (state.my - state.smy) * (1 - Math.exp(-dt * 3));

  if (dive) {
    const k = Math.min(1, (performance.now() - dive.t0) / dive.dur), e = k * k * k;
    camera.position.lerpVectors(dive.from, dive.target, e * .97);
    camera.fov = 55 - 40 * e; bloom.strength = 1.05 + 2.2 * e;
    look.lerp(dive.target, .2); camera.lookAt(look);
  } else {
    direct(t);
    const k = 1 - Math.exp(-dt0 * 2.2);
    camera.position.lerp(desired.pos, k); look.lerp(desired.look, k); camera.lookAt(look);
    camera.fov += (desired.fov - camera.fov) * k;
    bloom.strength += ((calm ? .7 : 1.05) - bloom.strength) * k;
  }
  camera.updateProjectionMatrix();
  const dimT = calm ? .45 : (state.mode === 'hero' && innerWidth < 900 ? .55 : 1);
  U.uDim.value += (dimT - U.uDim.value) * (1 - Math.exp(-dt0 * 2));

  // packets
  for (let i = 0; i < PK; i++) {
    const q = packets[i]; q.t += dt * (reduce ? .3 : .9);
    if (q.t >= 1) {
      nodes[q.b].lit = 1; q.t = 0; q.a = q.b; const nb = nodes[q.a].nb; q.b = nb[Math.floor(rnd() * nb.length)];
    }
    tmp.lerpVectors(nodes[q.a].p, nodes[q.b].p, q.t); pkPos.set([tmp.x, tmp.y, tmp.z], i * 3);
  }
  pkGeo.attributes.position.needsUpdate = true;
  // story spotlight
  if (state.mode === 'story') { storyNew.lit = Math.max(storyNew.lit, Math.min(1, U.uStory.value - .5)); storyOld.lit = Math.max(storyOld.lit, U.uStory.value < 2 ? .6 : 0); }
  for (let i = 0; i < N; i++) { nodes[i].lit *= Math.exp(-dt * 1.6); nLit[i] = nodes[i].lit; }
  litAttr.needsUpdate = true;
  edges.forEach(([a, b], k) => {
    const la = .16 + nodes[a].lit * .9, lb = .16 + nodes[b].lit * .9;
    const sa = nodes[a].stale * U.uStaleFade.value, sb = nodes[b].stale * U.uStaleFade.value;
    eCol.set([.95 * la * (1 - sa), .72 * la * (1 - sa), .42 * la * (1 - sa), .95 * lb * (1 - sb), .72 * lb * (1 - sb), .42 * lb * (1 - sb)], k * 6);
  });
  eColAttr.needsUpdate = true;
  rings.forEach((r, i) => { r.rotation.y += dt * (.05 + i * .02); });

  // pipeline gates
  const pipeOn = state.mode === 'pipeline' ? 1 : 0;
  stMat.uniforms.uOn.value += (pipeOn - stMat.uniforms.uOn.value) * (1 - Math.exp(-dt0 * 3));
  gates.forEach((g, i) => {
    const target = pipeOn && state.p * 6.6 >= i - .3 ? 1 : 0;
    g.lit += (target - g.lit) * (1 - Math.exp(-dt0 * 4));
    const vis = stMat.uniforms.uOn.value; g.t.material.color.copy(LILAC).lerp(AMBER, g.lit); g.t.material.opacity = (.3 + g.lit * .7) * vis; g.t.visible = vis > .01;
    g.t.rotation.z += dt * .1 * (i % 2 ? 1 : -1);
  });
  composer.render();
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false); composer.setSize(w, h);
  bloom.resolution.set(w / 2, h / 2);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();
addEventListener('pointermove', e => {
  state.mx = e.clientX / innerWidth * 2 - 1; state.my = -(e.clientY / innerHeight * 2 - 1);
  U.uMouse.value.set(state.mx, state.my);
}, { passive: true });
document.addEventListener('visibilitychange', () => { visible = !document.hidden; if (visible) clock.getDelta(); });

// Arrival: start inside a node and pull back out of it.
let arrived = false; try { arrived = sessionStorage.getItem('ev-zoom') === '1'; sessionStorage.removeItem('ev-zoom'); } catch (e) {}
direct(0);
if (arrived) {
  const n = nodes[Math.floor(N / 2)].p; camera.position.copy(n).add(new THREE.Vector3(0, 0, 1.2)); look.copy(n); camera.fov = 18;
} else { camera.position.copy(desired.pos).add(new THREE.Vector3(0, 8, 40)); look.copy(desired.look); }
camera.updateProjectionMatrix();
frame();

/* ---------- public API for the UI layer ---------- */
window.EV = Object.assign(window.EV || {}, {
  sceneReady: true,
  setMode(mode, p = 0) { state.mode = mode; state.p = p; },
  setProgress(p) { state.p = p; },
  setStory(step, staleFade) { U.uStory.value = step; U.uStaleFade.value = staleFade; },
  pulse(n = 6) { for (let i = 0; i < n; i++) nodes[Math.floor(rnd() * N)].lit = 1; },
  dive(duration = 900) {
    // fly into the node nearest the centre of the screen
    let best = null, bd = 1e9; const v = new THREE.Vector3();
    nodes.forEach(o => { v.copy(o.p).project(camera); if (v.z < 1 && v.z > -1) { const d = v.x * v.x + v.y * v.y; if (d < bd) { bd = d; best = o.p; } } });
    if (!best) best = S0;
    dive = { from: camera.position.clone(), target: best.clone(), t0: performance.now(), dur: duration };
    return new Promise(r => setTimeout(r, duration));
  }
});
document.dispatchEvent(new Event('ev:scene'));
