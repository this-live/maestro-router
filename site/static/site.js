/* Maestro website: sparkles, fade-in sections and copy buttons. No tracking, no cookies, no network requests. */
"use strict";
document.documentElement.classList.add("js");
(function () {
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* sparkle background (off when the reader prefers reduced motion) */
  var canvas = document.getElementById("sparkle");
  if (canvas && canvas.getContext && !reduce) {
    var ctx = canvas.getContext("2d"), W = 0, H = 0, mx = -100, my = -100, parts = [], frame = 0, running = true;
    var resize = function () { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; };
    resize();
    window.addEventListener("resize", resize);
    document.addEventListener("mousemove", function (e) { mx = e.clientX; my = e.clientY; });
    document.addEventListener("visibilitychange", function () { running = !document.hidden; if (running) requestAnimationFrame(loop); });
    var P = function (x, y, ambient) {
      this.x = x; this.y = y; this.ambient = ambient;
      this.vx = (Math.random() - 0.5) * (ambient ? 0.3 : 1.2);
      this.vy = (Math.random() - 0.5) * (ambient ? 0.3 : 1.2) - (ambient ? 0 : 0.5);
      this.size = Math.random() * 1.5 + 0.5;
      this.life = this.max = ambient ? Math.random() * 200 + 100 : Math.random() * 80 + 40;
    };
    P.prototype.step = function () {
      this.x += this.vx; this.y += this.vy; this.life--;
      if (this.ambient && this.life <= 0) { this.x = Math.random() * W; this.y = Math.random() * H; this.life = this.max; }
      var a = Math.min(1, this.life / this.max) * (this.ambient ? 0.25 : 0.6);
      ctx.beginPath(); ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(0,212,255," + a + ")"; ctx.shadowBlur = this.size * 4; ctx.shadowColor = "rgba(0,212,255,0.4)"; ctx.fill();
    };
    for (var i = 0; i < 40; i++) parts.push(new P(Math.random() * W, Math.random() * H, true));
    var loop = function () {
      if (!running) return;
      ctx.clearRect(0, 0, W, H); ctx.shadowBlur = 0;
      if (frame % 3 === 0 && mx > 0 && parts.length < 80) parts.push(new P(mx + (Math.random() - 0.5) * 8, my + (Math.random() - 0.5) * 8, false));
      for (var j = parts.length - 1; j >= 0; j--) { parts[j].step(); if (!parts[j].ambient && parts[j].life <= 0) parts.splice(j, 1); }
      frame++;
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  /* fade-in sections */
  var sections = document.querySelectorAll(".fade");
  if (reduce || !("IntersectionObserver" in window)) {
    sections.forEach(function (s) { s.classList.add("visible"); });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("visible"); io.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    sections.forEach(function (s) { io.observe(s); });
  }

  /* copy buttons: copy the <pre> next to them, without the shell prompt ("$ ") and sample output */
  document.querySelectorAll(".code").forEach(function (box) {
    var pre = box.querySelector("pre");
    if (!pre) return;
    var b = document.createElement("button");
    b.type = "button"; b.className = "copy"; b.textContent = "Copy";
    b.setAttribute("aria-label", "Copy to clipboard");
    b.addEventListener("click", function () {
      var clone = pre.cloneNode(true);
      clone.querySelectorAll(".p, .o").forEach(function (n) { n.remove(); });
      var text = clone.textContent.replace(/\n{2,}/g, "\n").replace(/^\n+|\n+$/g, "");
      var done = function () { b.textContent = "Copied"; b.classList.add("done"); setTimeout(function () { b.textContent = "Copy"; b.classList.remove("done"); }, 1500); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () { fallback(text); done(); });
      else { fallback(text); done(); }
    });
    var bar = document.createElement("div");
    bar.className = "code-bar";
    var title = box.previousElementSibling;
    if (title && title.classList.contains("code-title")) bar.appendChild(title);
    bar.appendChild(b);
    box.insertBefore(bar, pre);
  });
  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.setAttribute("readonly", ""); ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing else to try */ }
    ta.remove();
  }
})();
