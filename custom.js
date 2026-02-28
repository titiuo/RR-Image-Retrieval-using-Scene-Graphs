(function () {
  "use strict";

  const FALLBACK_OBJECTS = [
    "airplane","animal","arm","bag","banana","bench","bicycle","bird",
    "boat","book","bottle","bus","car","cat","chair","clock","coat",
    "cup","desk","dog","door","fence","flower","food","fork","fruit",
    "girl","glass","glove","grass","hand","hat","head","horse","house",
    "jacket","jeans","kite","knife","lamp","laptop","leaf","leg","man",
    "mirror","motorcycle","mountain","neck","pants","paper","person",
    "phone","pillow","pizza","plane","plate","player","pole","pot",
    "road","rock","shirt","shoe","shorts","sidewalk","sign","sky",
    "snow","sock","street","sun","table","tie","tire","train","tree",
    "truck","umbrella","vase","wall","water","wheel","window","woman"
  ];
  const FALLBACK_RELATIONS = [
    "above","attached to","behind","below","beside","between","carrying",
    "covered by","covering","eating","flying in","for","from","growing on",
    "hanging from","has","holding","in","in front of","inside","looking at",
    "lying on","made of","mounted on","near","of","on","on back of","over",
    "painted on","parked on","part of","playing","riding","sitting on",
    "standing on","to","under","using","walking in","walking on","watching",
    "wearing","wired to","with"
  ];
  const FALLBACK_ATTRIBUTES = [
    "adult","bare","beige","black","blue","brown","circular","clean",
    "colorful","dark","dry","empty","female","full","glass","gray",
    "green","large","left","light","long","male","metal","narrow",
    "old","open","orange","parked","pink","plastic","purple","red",
    "right","round","short","small","square","standing","striped",
    "tall","tan","thin","vertical","white","wide","wooden","yellow","young"
  ];

  let objects       = [];   // ["man", "car"]
  let relationships = [];   // [[0, "on", 1], ...]
  let attributes    = {};   // {0: ["black", "tall"], 1: [...]}

  let OBJ_VOCAB  = [];
  let REL_VOCAB  = [];
  let ATTR_VOCAB = [];

  /* ── Vocab loading ── */
  function loadVocab() {
    const tryFetch = (url, fallback) =>
      fetch(url)
        .then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
        .catch(() => fallback);

    Promise.all([
      tryFetch("object_classes.json",   FALLBACK_OBJECTS),
      tryFetch("relation_classes.json", FALLBACK_RELATIONS),
      tryFetch("attribute_classes.json",FALLBACK_ATTRIBUTES),
    ]).then(([objs, rels, attrs]) => {
      OBJ_VOCAB  = Array.isArray(objs)  ? objs.sort()  : FALLBACK_OBJECTS;
      REL_VOCAB  = Array.isArray(rels)  ? rels.sort()  : FALLBACK_RELATIONS;
      ATTR_VOCAB = Array.isArray(attrs) ? attrs.sort() : FALLBACK_ATTRIBUTES;
      injectBuilder();
    });
  }

  /* ── Inject UI ── */
  function injectBuilder() {
    const anchor =
      document.querySelector(".params_section") ||
      document.querySelector("#params")         ||
      document.querySelector("form");

    if (!anchor) { setTimeout(injectBuilder, 500); return; }

    const root = document.createElement("div");
    root.id = "sg-builder";
    root.innerHTML = builderHTML();
    anchor.parentNode.insertBefore(root, anchor);

    applyStyles();
    bindEvents();
    render();
    hookRunButton();
  }

  /* ── HTML ── */
  function builderHTML() {
    return `
<div class="sgb-wrap">
  <div class="sgb-header">
    <span class="sgb-title">⬡ Scene Graph Query Builder</span>
    <span class="sgb-sub">Select objects, attributes &amp; relationships — then click Run</span>
  </div>

  <div class="sgb-cols">

    <!-- ① OBJECTS -->
    <div class="sgb-panel">
      <div class="sgb-panel-label">① Objects</div>
      <select id="sgb-obj-select" size="7"></select>
      <button type="button" id="sgb-obj-add">➕ Add object</button>
      <div class="sgb-tags-label">Added:</div>
      <div id="sgb-obj-tags" class="sgb-tags"></div>
    </div>

    <!-- ② ATTRIBUTES -->
    <div class="sgb-panel">
      <div class="sgb-panel-label">② Attributes <span class="sgb-optional">(optional)</span></div>
      <div class="sgb-mini-label">Apply to object:</div>
      <select id="sgb-attr-target"></select>
      <select id="sgb-attr-select" size="7"></select>
      <button type="button" id="sgb-attr-add">➕ Add attribute</button>
      <div class="sgb-tags-label">Added:</div>
      <div id="sgb-attr-tags" class="sgb-tags"></div>
    </div>

    <!-- ③ RELATIONSHIPS -->
    <div class="sgb-panel">
      <div class="sgb-panel-label">③ Relationships <span class="sgb-optional">(optional)</span></div>
      <div class="sgb-rel-row">
        <div class="sgb-rel-col">
          <div class="sgb-mini-label">From</div>
          <select id="sgb-subj-select"></select>
        </div>
        <div class="sgb-rel-col">
          <div class="sgb-mini-label">Predicate</div>
          <select id="sgb-pred-select" size="7"></select>
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

    <!-- ④ PHRASE PREVIEW -->
    <div class="sgb-panel">
      <div class="sgb-panel-label">④ Your Query</div>
      <div id="sgb-phrase" class="sgb-phrase"></div>
      <button type="button" id="sgb-clear">🗑 Clear all</button>
    </div>

  </div>
  <div id="sgb-error" class="sgb-error" style="display:none"></div>
</div>`;
  }

  /* ── Styles ── */
  function applyStyles() {
    const s = document.createElement("style");
    s.textContent = `
#sg-builder { margin-bottom: 24px; }
.sgb-wrap {
  background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
  padding: 20px 24px; color: #e2e8f0;
  font-family: "IBM Plex Sans","Segoe UI",Arial,sans-serif; font-size: 13px;
}
.sgb-header {
  display: flex; align-items: baseline; gap: 14px;
  border-bottom: 1px solid #2d3748; padding-bottom: 12px; margin-bottom: 18px;
}
.sgb-title { font-size: 1rem; font-weight: 700; color: #90cdf4; }
.sgb-sub   { font-size: 0.72rem; color: #718096; text-transform: uppercase; letter-spacing:.07em; }
.sgb-optional { font-size: 0.6rem; color: #4a5568; font-weight: 400; }

.sgb-cols { display: flex; gap: 14px; flex-wrap: wrap; }
.sgb-panel {
  flex: 1; min-width: 180px; background: #111827;
  border: 1px solid #2d3748; border-radius: 8px;
  padding: 12px; display: flex; flex-direction: column; gap: 7px;
}
.sgb-panel-label {
  font-size: 0.66rem; font-weight: 700; text-transform: uppercase;
  letter-spacing:.1em; color: #73FBFD; margin-bottom: 2px;
}
.sgb-mini-label { font-size: 0.62rem; color: #718096; text-transform: uppercase; letter-spacing:.07em; }

.sgb-panel select {
  background: #1a2535; color: #e2e8f0; border: 1px solid #2d3748;
  border-radius: 5px; padding: 3px 5px; font-size: 12px; width: 100%;
  cursor: pointer; outline: none;
}
.sgb-panel select:focus { border-color: #63b3ed; }
.sgb-panel select option { background: #1a2535; }
.sgb-panel select[size="7"] { height: 140px; }

.sgb-panel button {
  background: #1a2535; color: #90cdf4; border: 1px solid #2b4070;
  border-radius: 5px; padding: 6px 10px; font-size: 12px; cursor: pointer;
  transition: all .15s; width: 100%; text-align: left;
}
.sgb-panel button:hover { background: #243352; border-color: #63b3ed; color: #bee3f8; }

.sgb-tags-label { font-size: 0.6rem; color: #718096; text-transform: uppercase; letter-spacing:.07em; }
.sgb-tags { display: flex; flex-direction: column; gap: 4px; min-height: 20px; }

.sgb-tag {
  display: flex; align-items: center; justify-content: space-between;
  border-radius: 4px; padding: 3px 8px; font-size: 11px; gap: 5px;
}
.sgb-tag-obj  { color: #90cdf4; border: 1px solid #2c5282; background: #162032; }
.sgb-tag-attr { color: #9ae6b4; border: 1px solid #22543d; background: #0f2a1a; }
.sgb-tag-rel  { color: #fbd38d; border: 1px solid #7b341e; background: #2a1508; }
.sgb-tag-rm {
  background: none; border: none; color: #4a5568; cursor: pointer;
  font-size: 11px; padding: 0 2px; line-height: 1; width: auto; flex-shrink: 0;
}
.sgb-tag-rm:hover { color: #fc8181; }

.sgb-phrase {
  background: #0f1117; border: 1px solid #1f2937; border-radius: 6px;
  padding: 12px 14px; font-family: "IBM Plex Mono",monospace; font-size: 12px;
  color: #718096; min-height: 80px; line-height: 1.9; flex: 1;
}
.sgb-ph-obj   { color: #90cdf4; font-weight: 600; }
.sgb-ph-attr  { color: #9ae6b4; }
.sgb-ph-pred  { color: #fbd38d; }
.sgb-ph-arrow { color: #4a5568; }
.sgb-ph-empty { color: #2d3748; font-style: italic; }

.sgb-rel-row { display: flex; gap: 7px; }
.sgb-rel-col { flex: 1; display: flex; flex-direction: column; gap: 4px; }

#sgb-clear {
  margin-top: auto; background: transparent;
  border-color: #2d3748; color: #718096; font-size: 11px;
}
#sgb-clear:hover { border-color: #e53e3e; color: #fc8181; background: #2d1515; }
.sgb-error {
  margin-top: 10px; padding: 8px 12px; background: #2d1515;
  border: 1px solid #e53e3e; border-radius: 5px; color: #fc8181; font-size: 12px;
}
    `;
    document.head.appendChild(s);
  }

  /* ── Render ── */
  function populateSelect(sel, items) {
    sel.innerHTML = items.map(v =>
      `<option value="${v}">${v}</option>`
    ).join("");
  }

  function render() {
    populateSelect(document.getElementById("sgb-obj-select"),  OBJ_VOCAB);
    populateSelect(document.getElementById("sgb-attr-select"), ATTR_VOCAB);
    populateSelect(document.getElementById("sgb-pred-select"), REL_VOCAB);

    /* Subject / Object dropdowns from current object list */
    const subjSel = document.getElementById("sgb-subj-select");
    const obj2Sel = document.getElementById("sgb-obj2-select");
    const attrTgt = document.getElementById("sgb-attr-target");
    const objOpts = objects.length === 0
      ? `<option value="" disabled>— add objects first —</option>`
      : objects.map((o, i) => `<option value="${i}">${i}: ${o}</option>`).join("");
    subjSel.innerHTML = objOpts;
    obj2Sel.innerHTML = objOpts;
    attrTgt.innerHTML = objOpts;

    /* Object tags */
    const objTags = document.getElementById("sgb-obj-tags");
    objTags.innerHTML = objects.length === 0
      ? `<span style="color:#2d3748;font-style:italic;font-size:11px">none yet</span>`
      : objects.map((o, i) =>
          `<div class="sgb-tag sgb-tag-obj">
            <span>⬡ ${i}: ${o}</span>
            <button class="sgb-tag-rm" data-rm-obj="${i}">✕</button>
          </div>`).join("");

    /* Attribute tags — grouped by object */
    const attrTags = document.getElementById("sgb-attr-tags");
    const attrEntries = Object.entries(attributes).flatMap(([i, attrs]) =>
      attrs.map(a => ({ i: parseInt(i), a }))
    );
    attrTags.innerHTML = attrEntries.length === 0
      ? `<span style="color:#2d3748;font-style:italic;font-size:11px">none yet</span>`
      : attrEntries.map(({ i, a }, idx) =>
          `<div class="sgb-tag sgb-tag-attr">
            <span>◈ <b>${objects[i]}</b>: ${a}</span>
            <button class="sgb-tag-rm" data-rm-attr-obj="${i}" data-rm-attr-val="${a}">✕</button>
          </div>`).join("");

    /* Relationship tags */
    const relTags = document.getElementById("sgb-rel-tags");
    relTags.innerHTML = relationships.length === 0
      ? `<span style="color:#2d3748;font-style:italic;font-size:11px">none yet</span>`
      : relationships.map(([s, pred, o], i) =>
          `<div class="sgb-tag sgb-tag-rel">
            <span>
              <span class="sgb-ph-obj">${objects[s]}</span>
              <span class="sgb-ph-arrow"> ── </span>
              <span class="sgb-ph-pred">[${pred}]</span>
              <span class="sgb-ph-arrow"> ──▶ </span>
              <span class="sgb-ph-obj">${objects[o]}</span>
            </span>
            <button class="sgb-tag-rm" data-rm-rel="${i}">✕</button>
          </div>`).join("");

    /* Phrase preview */
    const phrase = document.getElementById("sgb-phrase");
    if (objects.length === 0) {
      phrase.innerHTML = `<span class="sgb-ph-empty">← Add objects to build your query</span>`;
    } else {
      const objLine = objects.map((o, i) => {
        const objAttrs = (attributes[i] || []);
        const attrHtml = objAttrs.map(a =>
          ` <span class="sgb-ph-attr">[${a}]</span>`).join("");
        return `<span class="sgb-ph-obj">${o}</span>${attrHtml}`;
      }).join(` <span class="sgb-ph-arrow">·</span> `);

      const relLines = relationships.map(([s, pred, o]) =>
        `<br><span class="sgb-ph-arrow">  ↳ </span>` +
        `<span class="sgb-ph-obj">${objects[s]}</span>` +
        `<span class="sgb-ph-arrow"> ── </span>` +
        `<span class="sgb-ph-pred">[${pred}]</span>` +
        `<span class="sgb-ph-arrow"> ──▶ </span>` +
        `<span class="sgb-ph-obj">${objects[o]}</span>`
      ).join("");

      phrase.innerHTML = objLine + relLines;
    }

    syncIpolInputs();
  }

  /* ── Sync to IPOL inputs ──
     objects:       "man, car"
     relationships: "0 on 1, 0 next_to 2"   ← MUST include indices
  ── */
  function syncIpolInputs() {
    const objInput = findIpolInput("objects");
    const relInput = findIpolInput("relationships");
    if (objInput) objInput.value = objects.join(", ");
    if (relInput) relInput.value = relationships
      .map(([s, pred, o]) => `${s} ${pred} ${o}`)   // "0 on 1"
      .join(", ");
  }

  function findIpolInput(id) {
    return (
      document.querySelector(`input[name="${id}"]`)    ||
      document.querySelector(`textarea[name="${id}"]`) ||
      document.querySelector(`#id_${id}`)              ||
      document.querySelector(`input[id="${id}"]`)      ||
      document.querySelector(`[data-param="${id}"]`)
    );
  }

  /* ── Hook Run button ── */
  function hookRunButton() {
    const tryHook = () => {
      const btn =
        document.querySelector("button.run")            ||
        document.querySelector("input[value='Run']")    ||
        document.querySelector("button[type='submit']") ||
        document.querySelector(".run_button button")    ||
        document.querySelector("#run-button");
      if (!btn) { setTimeout(tryHook, 400); return; }
      btn.addEventListener("click", e => {
        if (objects.length === 0) {
          e.preventDefault(); e.stopPropagation();
          showError("Please add at least one object before running.");
          return;
        }
        hideError();
        syncIpolInputs();
      }, true);
    };
    tryHook();
  }

  /* ── Events ── */
  function bindEvents() {
    /* Add object */
    document.getElementById("sgb-obj-add").addEventListener("click", () => {
      const val = document.getElementById("sgb-obj-select").value;
      if (val && !objects.includes(val)) { objects.push(val); render(); }
    });
    document.getElementById("sgb-obj-select").addEventListener("dblclick", () =>
      document.getElementById("sgb-obj-add").click());

    /* Add attribute */
    document.getElementById("sgb-attr-add").addEventListener("click", () => {
      const tgt  = parseInt(document.getElementById("sgb-attr-target").value, 10);
      const val  = document.getElementById("sgb-attr-select").value;
      if (isNaN(tgt) || !val) { showError("Select an object and an attribute."); return; }
      hideError();
      if (!attributes[tgt]) attributes[tgt] = [];
      if (!attributes[tgt].includes(val)) { attributes[tgt].push(val); render(); }
    });
    document.getElementById("sgb-attr-select").addEventListener("dblclick", () =>
      document.getElementById("sgb-attr-add").click());

    /* Add relationship */
    document.getElementById("sgb-rel-add").addEventListener("click", () => {
      const si   = parseInt(document.getElementById("sgb-subj-select").value, 10);
      const oi   = parseInt(document.getElementById("sgb-obj2-select").value, 10);
      const pred = document.getElementById("sgb-pred-select").value;
      if (isNaN(si) || isNaN(oi)) { showError("Select subject and object."); return; }
      if (si === oi)               { showError("Subject and object must differ."); return; }
      if (!pred)                   { showError("Select a predicate."); return; }
      hideError();
      relationships.push([si, pred, oi]);
      render();
    });
    document.getElementById("sgb-pred-select").addEventListener("dblclick", () =>
      document.getElementById("sgb-rel-add").click());

    /* Remove tags — delegated */
    document.getElementById("sgb-obj-tags").addEventListener("click", e => {
      const btn = e.target.closest("[data-rm-obj]");
      if (!btn) return;
      const i = parseInt(btn.dataset.rmObj, 10);
      objects.splice(i, 1);
      // Re-index attributes
      const newAttrs = {};
      Object.entries(attributes).forEach(([k, v]) => {
        const ki = parseInt(k);
        if (ki === i) return;
        newAttrs[ki > i ? ki - 1 : ki] = v;
      });
      attributes = newAttrs;
      // Re-index / filter relationships
      relationships = relationships
        .filter(([s, , o]) => s !== i && o !== i)
        .map(([s, pred, o]) => [s > i ? s-1 : s, pred, o > i ? o-1 : o]);
      render();
    });

    document.getElementById("sgb-attr-tags").addEventListener("click", e => {
      const btn = e.target.closest("[data-rm-attr-val]");
      if (!btn) return;
      const oi  = parseInt(btn.dataset.rmAttrObj, 10);
      const val = btn.dataset.rmAttrVal;
      if (attributes[oi]) {
        attributes[oi] = attributes[oi].filter(a => a !== val);
        if (attributes[oi].length === 0) delete attributes[oi];
      }
      render();
    });

    document.getElementById("sgb-rel-tags").addEventListener("click", e => {
      const btn = e.target.closest("[data-rm-rel]");
      if (!btn) return;
      relationships.splice(parseInt(btn.dataset.rmRel, 10), 1);
      render();
    });

    /* Clear */
    document.getElementById("sgb-clear").addEventListener("click", () => {
      objects = []; attributes = {}; relationships = [];
      render();
    });
  }

  function showError(msg) {
    const el = document.getElementById("sgb-error");
    el.textContent = "⚠ " + msg; el.style.display = "block";
  }
  function hideError() {
    document.getElementById("sgb-error").style.display = "none";
  }

  /* ── Boot ── */
  if (document.readyState === "loading")
    document.addEventListener("DOMContentLoaded", loadVocab);
  else
    loadVocab();

})();