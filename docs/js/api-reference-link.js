document.addEventListener("DOMContentLoaded", () => {
  const isApiReferencePath = (pathname) => {
    const normalized = pathname.replace(/\/+$/, "");
    return normalized === "/reference" || normalized.includes("/reference/");
  };

  for (const link of document.querySelectorAll('a[href]')) {
    const href = link.getAttribute("href");
    if (!href) continue;

    let url;
    try {
      url = new URL(href, window.location.href);
    } catch {
      continue;
    }

    if (!isApiReferencePath(url.pathname)) continue;
    link.target = "_blank";
    link.rel = "noopener noreferrer";

    if (link.textContent?.trim() !== "API Reference") continue;

    link.classList.add("api-reference-external-link");
    link.setAttribute("aria-label", "API Reference (opens in a new tab)");

    if (link.querySelector(".api-reference-external-icon")) continue;

    const icon = document.createElement("span");
    icon.className = "api-reference-external-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = " ↗";
    link.appendChild(icon);
  }
});
