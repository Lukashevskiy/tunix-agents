/* Interactive task map: Cytoscape supplies pan, zoom, drag and force-directed layout. */
(function () {
  function renderTaskGraph() {
    var container = document.querySelector("#task-graph");
    var dataElement = document.querySelector("#task-graph-data");
    if (!container || !dataElement || container.dataset.rendered) return;
    if (typeof cytoscape === "undefined") {
      container.textContent = "Не удалось загрузить библиотеку графа. Проверьте сетевое подключение и перезагрузите страницу.";
      return;
    }

    var payload = JSON.parse(dataElement.textContent);
    var detail = document.querySelector("#task-graph-detail");
    var cy = cytoscape({
      container: container,
      elements: payload.elements,
      wheelSensitivity: 0.18,
      style: [
        { selector: "node[type = 'task']", style: {
          "background-color": "data(color)", "border-color": "data(border)", "border-width": 2,
          "color": "#ffffff", "font-size": 10, "label": "data(label)", "shape": "round-rectangle",
          "text-max-width": 130, "text-wrap": "wrap", "text-valign": "center", "text-halign": "center",
          "width": 145, "height": 52
        } },
        { selector: "node[type = 'module']", style: {
          "background-color": "#8b949e", "background-opacity": 0.08, "border-color": "#8b949e",
          "border-width": 1, "color": "#8b949e", "font-size": 12, "font-weight": "bold",
          "label": "data(label)", "padding": 24, "text-valign": "top", "text-halign": "center"
        } },
        { selector: "edge", style: {
          "curve-style": "bezier", "line-color": "#8b949e", "target-arrow-color": "#8b949e",
          "target-arrow-shape": "triangle", "width": 1.5, "opacity": 0.72
        } },
        { selector: ".selected", style: { "border-width": 4, "border-color": "#f0f6fc" } }
      ],
      layout: { name: "cose", animate: false, padding: 38, nodeRepulsion: 900000, idealEdgeLength: 95 }
    });

    function showDetail(node) {
      cy.nodes().removeClass("selected");
      node.addClass("selected");
      var task = node.data();
      var heading = document.createElement("h2");
      heading.textContent = task.title;
      var description = document.createElement("p");
      description.textContent = task.description;
      var metadata = document.createElement("p");
      metadata.textContent = "Модуль: " + task.module + " · Статус: " + task.statusLabel;
      var previous = document.createElement("p");
      previous.textContent = "Предшествующая задача: " + (task.previous || "нет");
      detail.replaceChildren(heading, description, metadata, previous);
    }

    cy.on("tap", "node[type = 'task']", function (event) { showDetail(event.target); });
    document.querySelector("#task-graph-fit").addEventListener("click", function () { cy.fit(undefined, 35); });
    document.querySelector("#task-graph-layout").addEventListener("click", function () {
      cy.layout({ name: "cose", animate: true, padding: 38, nodeRepulsion: 900000, idealEdgeLength: 95 }).run();
    });
    container.dataset.rendered = "true";
  }

  if (typeof document$ !== "undefined") document$.subscribe(renderTaskGraph);
  else document.addEventListener("DOMContentLoaded", renderTaskGraph);
}());
