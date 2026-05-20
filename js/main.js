/* ================================================================
   main.js — Navigation behaviour for hakeemshindy.github.io
   ================================================================ */

(function () {
  'use strict';

  /* ── Smooth-scroll for anchor links ──────────────────────────── */
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      const id = this.getAttribute('href');
      if (id === '#') return;
      const target = document.querySelector(id);
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      // Close mobile menu if open
      closeMobileMenu();
    });
  });

  /* ── Active-section highlight via IntersectionObserver ───────── */
  const navLinks = document.querySelectorAll('#nav-links a[href^="#"]');

  const sectionIds = Array.from(navLinks).map(function (a) {
    return a.getAttribute('href').replace('#', '');
  }).filter(Boolean);

  const sections = sectionIds.map(function (id) {
    return document.getElementById(id);
  }).filter(Boolean);

  // Keep track of which sections are currently intersecting
  const visible = new Set();

  const observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (entry.isIntersecting) {
        visible.add(entry.target.id);
      } else {
        visible.delete(entry.target.id);
      }
    });
    updateActiveLink();
  }, {
    rootMargin: '-56px 0px -40% 0px', // offset for sticky nav height
    threshold: 0
  });

  sections.forEach(function (sec) { observer.observe(sec); });

  function updateActiveLink() {
    // Pick the topmost visible section in document order
    let activeId = null;
    for (var i = 0; i < sections.length; i++) {
      if (visible.has(sections[i].id)) {
        activeId = sections[i].id;
        break;
      }
    }
    navLinks.forEach(function (a) {
      if (a.getAttribute('href') === '#' + activeId) {
        a.classList.add('active');
      } else {
        a.classList.remove('active');
      }
    });
  }

  /* ── Mobile hamburger toggle ─────────────────────────────────── */
  const hamburger = document.getElementById('hamburger');
  const navMenu   = document.getElementById('nav-links');

  if (hamburger && navMenu) {
    hamburger.addEventListener('click', function () {
      const isOpen = navMenu.classList.toggle('open');
      hamburger.classList.toggle('open', isOpen);
      hamburger.setAttribute('aria-expanded', String(isOpen));
    });

    // Close on outside click
    document.addEventListener('click', function (e) {
      if (!hamburger.contains(e.target) && !navMenu.contains(e.target)) {
        closeMobileMenu();
      }
    });
  }

  function closeMobileMenu() {
    if (navMenu)   navMenu.classList.remove('open');
    if (hamburger) {
      hamburger.classList.remove('open');
      hamburger.setAttribute('aria-expanded', 'false');
    }
  }

})();
