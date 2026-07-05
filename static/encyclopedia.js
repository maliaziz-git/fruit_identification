(function () {
  const searchInput = document.getElementById("fruit-search");
  const emptyState = document.getElementById("enc-empty");
  const cards = Array.from(document.querySelectorAll("[data-fruit-card]"));

  function filterCards(query) {
    const q = query.trim().toLowerCase();
    let visible = 0;
    cards.forEach((card) => {
      const haystack = (card.getAttribute("data-search") || "").toLowerCase();
      const show = !q || haystack.includes(q);
      card.classList.toggle("hidden", !show);
      if (show) visible += 1;
    });
    emptyState.classList.toggle("hidden", visible > 0);
  }

  if (searchInput) {
    searchInput.addEventListener("input", () => filterCards(searchInput.value));
  }

  document.querySelectorAll("[data-fruit-img]").forEach((img) => {
    img.addEventListener("error", () => {
      img.classList.add("hidden");
      const placeholder = img.parentElement?.querySelector("[data-fruit-placeholder]");
      if (placeholder) placeholder.classList.remove("hidden");
    });
  });

  document.querySelectorAll(".enc-card").forEach((card) => {
    const tabs = card.querySelectorAll(".enc-tab");
    const panels = card.querySelectorAll(".enc-panel");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const name = tab.getAttribute("data-tab");
        tabs.forEach((t) => {
          const active = t === tab;
          t.classList.toggle("active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
        panels.forEach((panel) => {
          const show = panel.getAttribute("data-panel") === name;
          panel.classList.toggle("active", show);
          panel.hidden = !show;
        });
      });
    });
  });
})();
