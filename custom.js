/**
 * custom.js — Scene Graph Query Builder for IPOL
 *
 * Injects an interactive builder above the native IPOL params panel.
 * When the user clicks Run, it writes the comma-separated objects string
 * and relationship triplets into the hidden IPOL text inputs so the
 * demo.json "run" command receives them correctly.
 *
 * Vocabulary is loaded from the two JSON files baked into the repo
 * at /data/object_classes.json and /data/relation_classes.json.
 * Fallback built-in lists are used if fetching fails.
 */

(function () {
  "use strict";

  /* ── Fallback mini-vocabulary (used if JSON fetch fails) ── */
  const FALLBACK_OBJECTS = [
    "airplane",
    "animal",
    "arm",
    "bag",
    "banana",
    "bench",
    "bicycle",
    "bird",
    "boat",
    "book",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "clock",
    "coat",
    "cup",
    "desk",
    "dog",
    "door",
    "fence",
    "flower",
    "food",
    "fork",
    "fruit",
    "girl",
    "glass",
    "glove",
    "grass",
    "hand",
    "hat",
    "head",
    "horse",
    "house",
    "jacket",
    "jeans",
    "kite",
    "knife",
    "lamp",
    "laptop",
    "leaf",
    "leg",
    "man",
    "mirror",
    "motorcycle",
    "mountain",
    "neck",
    "pants",
    "paper",
    "person",
    "phone",
    "pillow",
    "pizza",
    "plane",
    "plate",
    "player",
    "pole",
    "pot",
    "road",
    "rock",
    "shirt",
    "shoe",
    "shorts",
    "sidewalk",
    "sign",
    "sky",
    "snow",
    "sock",
    "street",
    "sun",
    "table",
    "tie",
    "tire",
    "train",
    "tree",
    "truck",
    "umbrella",
    "vase",
    "wall",
    "water",
    "wheel",
    "window",
    "woman",
  ];
  const FALLBACK_RELATIONS = [
    "above",
    "attached to",
    "behind",
    "below",
    "beside",
    "between",
    "carrying",
    "covered by",
    "covering",
    "eating",
    "flying in",
    "for",
    "from",
    "growing on",
    "hanging from",
    "has",
    "holding",
    "in",
    "in front of",
    "inside",
    "looking at",
    "lying on",
    "made of",
    "mounted on",
    "near",
    "of",
    "on",
    "on back of",
    "over",
    "painted on",
    "parked on",
    "part of",
    "playing",
    "riding",
    "says",
    "sitting on",
    "standing on",
    "to",
    "under",
    "using",
    "walking in",
    "walking on",
    "watching",
    "wearing",
    "wired to",
    "with",
  ];

  /* ── State ── */
  let objects = []; // ["person", "horse"]
  let relationships = []; // [[0, "on", 1], ...]

  let OBJ_VOCAB = [];
  let REL_VOCAB = [];

  /* ──────────────────────────────────────────────────────────
     Vocabulary loading
  ────────────────────────────────────────────────────────── */
  function loadVocab() {
    // Try to fetch from the repo's data dir served as static assets.
    // IPOL serves $bin as a static path at /api/core/... or similar —
    // if unavailable we fall back immediately to the built-in lists.
    const tryFetch = (url, fallback) =>
      fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(r.status);
          return r.json();
        })
        .catch(() => fallback);

    Promise.all([
      tryFetch("object_classes.json", FALLBACK_OBJECTS),
      tryFetch("relation_classes.json", FALLBACK_RELATIONS),
    ]).then(([objs, rels]) => {
      OBJ_VOCAB = Array.isArray(objs) ? objs.sort() : FALLBACK_OBJECTS;
      REL_VOCAB = Array.isArray(rels) ? rels.sort() : FALLBACK_RELATIONS;
      injectBuilder();
    });
  }

  /* ──────────────────────────────────────────────────────────
     UI injection
  ────────────────────────────────────────────────────────── */
  function injectBuilder() {
    /* Find the IPOL params section to insert before it */
    const anchor =
      document.querySelector(".params_section") ||
      document.querySelector("#params") ||
      document.querySelector("form");

    if (!anchor) {
      console.warn(
        "[SceneGraph] Could not find params anchor — retrying in 500ms",
      );
      setTimeout(injectBuilder, 500);
      return;
    }

    /* Build root container */
    const root = document.createElement("div");
    root.id = "sg-builder";
    root.innerHTML = builderHTML();
    anchor.parentNode.insertBefore(root, anchor);

    applyStyles();
    bindEvents();
    render();

    /* Hook into the Run button — write values into IPOL inputs */
    hookRunButton();
  }

  /* ──────────────────────────────────────────────────────────
     HTML template
  ────────────────────────────────────────────────────────── */
  function builderHTML() {
    return `
<div class="sgb-wrap">
  <div class="sgb-header">
    <span class="sgb-title">⬡ Scene Graph Query Builder</span>
    <span class="sgb-sub">Select objects &amp; relationships, then click Run</span>
  </div>

  <div class="sgb-cols">

    <!-- ① OBJECTS -->
    <div class="sgb-panel" id="sgb-panel-obj">
      <div class="sgb-panel-label">① Objects</div>
      <select id="sgb-obj-select" size="6"></select>
      <button type="button" id="sgb-obj-add">➕ Add object</button>
      <div class="sgb-tags-label">Added:</div>
      <div id="sgb-obj-tags" class="sgb-tags"></div>
    </div>

    <!-- ② RELATIONSHIPS -->
    <div class="sgb-panel" id="sgb-panel-rel">
      <div class="sgb-panel-label">② Relationships</div>
      <div class="sgb-rel-row">
        <div class="sgb-rel-col">
          <div class="sgb-mini-label">From</div>
          <select id="sgb-subj-select"></select>
        </div>
        <div class="sgb-rel-col">
          <div class="sgb-mini-label">Predicate</div>
          <select id="sgb-pred-select" size="6"></select>
        </div>
        <div class="sgb-rel-col">
          <div class="sgb-mini-label">To</div>
          <select id="sgb-obj2-select"></select>
        </div>
      </div>
      <button type="button" id="sgb-rel-add">➕ Add relationship</button>
      <div class="sgb-tags-label">Added:</div>
      <div id="sgb-rel-tags" class="sgb-tags"></div>
    </div>

    <!-- ③ PHRASE PREVIEW -->
    <div class="sgb-panel" id="sgb-panel-phrase">
      <div class="sgb-panel-label">③ Your Query</div>
      <div id="sgb-phrase" class="sgb-phrase"></div>
      <button type="button" id="sgb-clear">🗑 Clear all</button>
    </div>

  </div><!-- .sgb-cols -->

  <div id="sgb-error" class="sgb-error" style="display:none"></div>
</div><!-- .sgb-wrap -->`;
  }

  /* ──────────────────────────────────────────────────────────
     Styles
  ────────────────────────────────────────────────────────── */
  function applyStyles() {
    const s = document.createElement("style");
    s.textContent = `
#sg-builder { margin-bottom: 24px; }

.sgb-wrap {
  background: #1a1f2e;
  border: 1px solid #2d3748;
  border-radius: 10px;
  padding: 20px 24px;
  color: #e2e8f0;
  font-family: "IBM Plex Sans", "Segoe UI", Arial, sans-serif;
  font-size: 13px;
}

.sgb-header {
  display: flex;
  align-items: baseline;
  gap: 14px;
  border-bottom: 1px solid #2d3748;
  padding-bottom: 12px;
  margin-bottom: 18px;
}
.sgb-title {
  font-size: 1rem;
  font-weight: 700;
  color: #90cdf4;
  letter-spacing: -.01em;
}
.sgb-sub {
  font-size: 0.75rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: .08em;
}

.sgb-cols {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.sgb-panel {
  flex: 1;
  min-width: 200px;
  background: #111827;
  border: 1px solid #2d3748;
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.sgb-panel-label {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .1em;
  color: #73FBFD;
  margin-bottom: 4px;
}

.sgb-mini-label {
  font-size: 0.65rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 3px;
}

.sgb-panel select {
  background: #1a2535;
  color: #e2e8f0;
  border: 1px solid #2d3748;
  border-radius: 5px;
  padding: 4px 6px;
  font-size: 12px;
  width: 100%;
  cursor: pointer;
  outline: none;
}
.sgb-panel select:focus { border-color: #63b3ed; }
.sgb-panel select option { background: #1a2535; }
.sgb-panel select[size="6"] { height: 130px; }

.sgb-panel button {
  background: #1a2535;
  color: #90cdf4;
  border: 1px solid #2b4070;
  border-radius: 5px;
  padding: 6px 10px;
  font-size: 12px;
  cursor: pointer;
  transition: all .15s;
  width: 100%;
  text-align: left;
}
.sgb-panel button:hover {
  background: #243352;
  border-color: #63b3ed;
  color: #bee3f8;
}

.sgb-tags-label {
  font-size: 0.62rem;
  color: #718096;
  text-transform: uppercase;
  letter-spacing: .07em;
}

.sgb-tags {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-height: 24px;
}

.sgb-tag {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #1a2535;
  border: 1px solid #2b4070;
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 11px;
  gap: 6px;
}
.sgb-tag-obj  { color: #90cdf4; border-color: #2c5282; background: #162032; }
.sgb-tag-rel  { color: #fbd38d; border-color: #7b341e; background: #2a1508; }

.sgb-tag-rm {
  background: none;
  border: none;
  color: #4a5568;
  cursor: pointer;
  font-size: 11px;
  padding: 0 2px;
  line-height: 1;
  width: auto;
  flex-shrink: 0;
}
.sgb-tag-rm:hover { color: #fc8181; background: none; border: none; }

.sgb-phrase {
  background: #0f1117;
  border: 1px solid #1f2937;
  border-radius: 6px;
  padding: 12px 14px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 12px;
  color: #718096;
  min-height: 80px;
  line-height: 1.8;
  flex: 1;
}
.sgb-ph-obj   { color: #90cdf4; font-weight: 600; }
.sgb-ph-pred  { color: #fbd38d; }
.sgb-ph-arrow { color: #4a5568; }
.sgb-ph-empty { color: #2d3748; font-style: italic; }

.sgb-rel-row {
  display: flex;
  gap: 8px;
}
.sgb-rel-col { flex: 1; display: flex; flex-direction: column; }

#sgb-clear {
  margin-top: auto;
  background: transparent;
  border-color: #2d3748;
  color: #718096;
  font-size: 11px;
}
#sgb-clear:hover { border-color: #e53e3e; color: #fc8181; background: #2d1515; }

.sgb-error {
  margin-top: 10px;
  padding: 8px 12px;
  background: #2d1515;
  border: 1px solid #e53e3e;
  border-radius: 5px;
  color: #fc8181;
  font-size: 12px;
}
    `;
    document.head.appendChild(s);
  }

  /* ──────────────────────────────────────────────────────────
     Render
  ────────────────────────────────────────────────────────── */
  function populateSelect(sel, items, placeholder) {
    sel.innerHTML = "";
    if (placeholder) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = placeholder;
      opt.disabled = true;
      sel.appendChild(opt);
    }
    items.forEach((item) => {
      const opt = document.createElement("option");
      opt.value = item;
      opt.textContent = item;
      sel.appendChild(opt);
    });
  }

  function render() {
    /* Object vocabulary list */
    populateSelect(document.getElementById("sgb-obj-select"), OBJ_VOCAB);

    /* Predicate vocabulary list */
    populateSelect(document.getElementById("sgb-pred-select"), REL_VOCAB);

    /* Subject / Object dropdowns (populated from current objects) */
    const subjSel = document.getElementById("sgb-subj-select");
    const obj2Sel = document.getElementById("sgb-obj2-select");
    const objItems = objects.map((o, i) => `${i}: ${o}`);
    populateSelect(
      subjSel,
      objItems,
      objects.length === 0 ? "— add objects first —" : null,
    );
    populateSelect(
      obj2Sel,
      objItems,
      objects.length === 0 ? "— add objects first —" : null,
    );

    /* Object tags */
    const objTags = document.getElementById("sgb-obj-tags");
    objTags.innerHTML =
      objects.length === 0
        ? '<span style="color:#2d3748;font-style:italic;font-size:11px">none yet</span>'
        : objects
            .map(
              (o, i) => `
          <div class="sgb-tag sgb-tag-obj">
            <span>⬡ ${i}: ${o}</span>
            <button class="sgb-tag-rm" data-rm-obj="${i}" title="Remove">✕</button>
          </div>`,
            )
            .join("");

    /* Relationship tags */
    const relTags = document.getElementById("sgb-rel-tags");
    relTags.innerHTML =
      relationships.length === 0
        ? '<span style="color:#2d3748;font-style:italic;font-size:11px">none yet</span>'
        : relationships
            .map((r, i) => {
              const [s, pred, o] = r;
              const sName = objects[s] || s;
              const oName = objects[o] || o;
              return `
          <div class="sgb-tag sgb-tag-rel">
            <span>↔ <span class="sgb-ph-obj">${sName}</span>
              <span class="sgb-ph-arrow"> ── </span>
              <span class="sgb-ph-pred">[${pred}]</span>
              <span class="sgb-ph-arrow"> ──▶ </span>
              <span class="sgb-ph-obj">${oName}</span></span>
            <button class="sgb-tag-rm" data-rm-rel="${i}" title="Remove">✕</button>
          </div>`;
            })
            .join("");

    /* Phrase preview */
    const phrase = document.getElementById("sgb-phrase");
    if (objects.length === 0) {
      phrase.innerHTML =
        '<span class="sgb-ph-empty">← Add objects to build your query</span>';
    } else {
      const objLine = objects
        .map((o) => `<span class="sgb-ph-obj">${o}</span>`)
        .join(' <span class="sgb-ph-arrow">·</span> ');
      const relLines = relationships
        .map(
          ([s, pred, o]) =>
            `<br><span class="sgb-ph-arrow">  ↳ </span>` +
            `<span class="sgb-ph-obj">${objects[s]}</span>` +
            `<span class="sgb-ph-arrow"> ── </span>` +
            `<span class="sgb-ph-pred">[${pred}]</span>` +
            `<span class="sgb-ph-arrow"> ──▶ </span>` +
            `<span class="sgb-ph-obj">${objects[o]}</span>`,
        )
        .join("");
      phrase.innerHTML = objLine + relLines;
    }

    /* Sync hidden IPOL text inputs */
    syncIpolInputs();
  }

  /* ──────────────────────────────────────────────────────────
     Sync into IPOL inputs
  ────────────────────────────────────────────────────────── */
  function syncIpolInputs() {
    const objInput = findIpolInput("objects");
    const relInput = findIpolInput("relationships");
    if (objInput) objInput.value = objects.join(", ");
    if (relInput)
      relInput.value = relationships
        .map(([s, pred, o]) => `${s} ${pred} ${o}`)
        .join(", ");
  }

  function findIpolInput(paramId) {
    /* IPOL renders text params as <input> or <textarea> with name or id = param id */
    return (
      document.querySelector(`input[name="${paramId}"]`) ||
      document.querySelector(`textarea[name="${paramId}"]`) ||
      document.querySelector(`#id_${paramId}`) ||
      document.querySelector(`input[id="${paramId}"]`) ||
      document.querySelector(`[data-param="${paramId}"]`)
    );
  }

  /* ──────────────────────────────────────────────────────────
     Hook Run button
  ────────────────────────────────────────────────────────── */
  function hookRunButton() {
    /* Try to find and wrap the Run button */
    const tryHook = () => {
      const btn =
        document.querySelector("button.run") ||
        document.querySelector("input[value='Run']") ||
        document.querySelector("button[type='submit']") ||
        document.querySelector(".run_button button") ||
        document.querySelector("#run-button");

      if (!btn) {
        setTimeout(tryHook, 400);
        return;
      }

      btn.addEventListener(
        "click",
        function (e) {
          if (objects.length === 0) {
            e.preventDefault();
            e.stopPropagation();
            showError("Please add at least one object before running.");
            return;
          }
          hideError();
          syncIpolInputs(); /* ensure values are written right before submit */
        },
        true,
      );
    };
    tryHook();
  }

  /* ──────────────────────────────────────────────────────────
     Events
  ────────────────────────────────────────────────────────── */
  function bindEvents() {
    /* Add object */
    document.getElementById("sgb-obj-add").addEventListener("click", () => {
      const sel = document.getElementById("sgb-obj-select");
      const val = sel.value;
      if (!val) return;
      if (!objects.includes(val)) {
        objects.push(val);
        render();
      }
    });

    /* Double-click on obj list also adds */
    document
      .getElementById("sgb-obj-select")
      .addEventListener("dblclick", () => {
        document.getElementById("sgb-obj-add").click();
      });

    /* Add relationship */
    document.getElementById("sgb-rel-add").addEventListener("click", () => {
      const subjSel = document.getElementById("sgb-subj-select");
      const predSel = document.getElementById("sgb-pred-select");
      const obj2Sel = document.getElementById("sgb-obj2-select");

      const si = parseInt(subjSel.value, 10);
      const oi = parseInt(obj2Sel.value, 10);
      const pred = predSel.value;

      if (isNaN(si) || isNaN(oi)) {
        showError("Select both a subject and object for the relationship.");
        return;
      }
      if (si === oi) {
        showError("Subject and object must be different.");
        return;
      }
      if (!pred) {
        showError("Select a predicate.");
        return;
      }

      hideError();
      relationships.push([si, pred, oi]);
      render();
    });

    /* Double-click on pred list also adds */
    document
      .getElementById("sgb-pred-select")
      .addEventListener("dblclick", () => {
        document.getElementById("sgb-rel-add").click();
      });

    /* Remove tags (delegated) */
    document.getElementById("sgb-obj-tags").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-rm-obj]");
      if (!btn) return;
      const i = parseInt(btn.dataset.rmObj, 10);
      objects.splice(i, 1);
      /* Adjust relationship indices */
      relationships = relationships
        .filter(([s, , o]) => s !== i && o !== i)
        .map(([s, pred, o]) => [s > i ? s - 1 : s, pred, o > i ? o - 1 : o]);
      render();
    });

    document.getElementById("sgb-rel-tags").addEventListener("click", (e) => {
      const btn = e.target.closest("[data-rm-rel]");
      if (!btn) return;
      relationships.splice(parseInt(btn.dataset.rmRel, 10), 1);
      render();
    });

    /* Clear */
    document.getElementById("sgb-clear").addEventListener("click", () => {
      objects = [];
      relationships = [];
      render();
    });
  }

  function showError(msg) {
    const el = document.getElementById("sgb-error");
    el.textContent = "⚠ " + msg;
    el.style.display = "block";
  }
  function hideError() {
    document.getElementById("sgb-error").style.display = "none";
  }

  /* ──────────────────────────────────────────────────────────
     Boot
  ────────────────────────────────────────────────────────── */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", loadVocab);
  } else {
    loadVocab();
  }
})();
