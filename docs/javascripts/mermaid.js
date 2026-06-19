/* Render Mermaid after every Material page navigation, not only the initial load. */
document$.subscribe(function () {
  mermaid.initialize({ startOnLoad: false, securityLevel: "strict" });
  mermaid.run({ querySelector: ".mermaid" });
});
