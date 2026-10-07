// Counts at identical (X,Y) coordinates preserve every observation in the OLS fit.
(() => {
  "use strict";
  const svgNS = "http://www.w3.org/2000/svg";
  const format = (value, digits = 2) => value.toLocaleString("en-US", {
    minimumFractionDigits: digits, maximumFractionDigits: digits
  });

  function node(tag, attributes = {}, text) {
    const element = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function mergePoints(years, year) {
    const counts = new Map();
    for (const item of years) {
      if (year !== "all" && String(item.year) !== year) continue;
      for (const [x, y, n] of item.points) {
        const key = `${x},${y}`;
        const previous = counts.get(key);
        if (previous) previous[2] += n;
        else counts.set(key, [x, y, n]);
      }
    }
    return [...counts.values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }

  function fit(points) {
    const totals = { n: 0, x: 0, y: 0, xx: 0, xy: 0, yy: 0 };
    for (const [x, y, n] of points) {
      totals.n += n; totals.x += n*x; totals.y += n*y;
      totals.xx += n*x*x; totals.xy += n*x*y; totals.yy += n*y*y;
    }
    const sxx = totals.xx - totals.x*totals.x/totals.n;
    const sxy = totals.xy - totals.x*totals.y/totals.n;
    const slope = sxy/sxx;
    const intercept = (totals.y - slope*totals.x)/totals.n;
    const sst = totals.yy - totals.y*totals.y/totals.n;
    return { ...totals, intercept, slope, sst };
  }

  function squaredError(totals, intercept, slope) {
    return Math.max(0, totals.yy - 2*intercept*totals.y - 2*slope*totals.xy +
      totals.n*intercept*intercept + 2*intercept*slope*totals.x + slope*slope*totals.xx);
  }

  async function initialize(root) {
    if (root.dataset.initialized) return;
    root.dataset.initialized = "true";
    const status = root.querySelector("[data-status]");
    try {
      const response = await fetch(root.dataset.src);
      if (!response.ok) throw new Error(`Data request failed (${response.status})`);
      const data = await response.json();
      const control = (name) => root.querySelector(`[data-control="${name}"]`);
      const value = (name, text) => { root.querySelector(`[data-value="${name}"]`).textContent = text; };
      for (const item of data.years) {
        const option = document.createElement("option");
        option.value = String(item.year); option.textContent = String(item.year);
        control("year").append(option);
      }

      let points, fitted, intercept, slope, plottedExtent;
      const chart = root.querySelector("[data-chart]");
      const clipID = `${root.id}-clip`;
      const frame = { left: 58, right: 682, top: 18, bottom: 354 };
      const defs = node("defs");
      const clip = node("clipPath", { id: clipID });
      clip.append(node("rect", { x: frame.left, y: frame.top,
        width: frame.right-frame.left, height: frame.bottom-frame.top }));
      defs.append(clip); chart.append(defs);
      const axes = node("g");
      const observations = node("g", { "clip-path": `url(#${clipID})` });
      const residuals = node("g", { "clip-path": `url(#${clipID})` });
      const lines = node("g", { "clip-path": `url(#${clipID})` });
      chart.append(axes, observations, residuals, lines);

      function scaleX(x) {
        return frame.left + (frame.right-frame.left)*x/plottedExtent.x;
      }
      function scaleY(y) {
        return frame.bottom - (frame.bottom-frame.top)*(y-plottedExtent.low)/
          (plottedExtent.high-plottedExtent.low);
      }

      function drawChart(attempts) {
        const maxX = Math.max(...points.map(([x]) => x));
        const maxY = Math.max(...points.map(([, y]) => y));
        const endpoints = [intercept, intercept+slope*maxX,
          fitted.intercept, fitted.intercept+fitted.slope*maxX];
        const low = Math.min(0, Math.floor(Math.min(...endpoints)));
        const high = Math.ceil(Math.max(maxY, ...endpoints, 1) * 1.06);
        const extent = `${maxX},${low},${high}`;
        if (!plottedExtent || plottedExtent.key !== extent) {
          plottedExtent = { x: maxX, low, high, key: extent };
          axes.replaceChildren(); observations.replaceChildren();
          const xStep = maxX > 40 ? 20 : 10;
          for (let x = 0; x <= maxX; x += xStep) {
            axes.append(node("line", { x1: scaleX(x), x2: scaleX(x), y1: frame.top,
              y2: frame.bottom, class: "rp-grid" }));
            axes.append(node("text", { x: scaleX(x), y: frame.bottom+22,
              "text-anchor": "middle", class: "rp-axis" }, x));
          }
          const yStep = Math.max(1, Math.ceil((high-low)/6));
          for (let y = Math.ceil(low/yStep)*yStep; y <= high; y += yStep) {
            axes.append(node("line", { x1: frame.left, x2: frame.right,
              y1: scaleY(y), y2: scaleY(y), class: "rp-grid" }));
            axes.append(node("text", { x: frame.left-10, y: scaleY(y)+4,
              "text-anchor": "end", class: "rp-axis" }, y));
          }
          axes.append(node("text", { x: (frame.left+frame.right)/2, y: 402,
            "text-anchor": "middle", class: "rp-axis-label" }, "Attack attempts (X)"));
          axes.append(node("text", { transform: "translate(16 186) rotate(-90)",
            "text-anchor": "middle", class: "rp-axis-label" }, "Attack errors (Y)"));
          const maxCount = Math.max(...points.map(([, , n]) => n));
          for (const [x, y, count] of points) {
            const dot = node("circle", { cx: scaleX(x), cy: scaleY(y),
              r: 1.5+9*Math.sqrt(count/maxCount), class: "rp-dot" });
            dot.append(node("title", {}, `${x} attempts, ${y} errors: ${format(count, 0)} player match records`));
            observations.append(dot);
          }
        }
        residuals.replaceChildren();
        if (control("residuals").checked) {
          for (const [x, y] of points) {
            residuals.append(node("line", { x1: scaleX(x), x2: scaleX(x),
              y1: scaleY(y), y2: scaleY(intercept+slope*x), class: "rp-residual" }));
          }
        }
        lines.replaceChildren();
        lines.append(node("line", { x1: scaleX(0), x2: scaleX(maxX),
          y1: scaleY(intercept), y2: scaleY(intercept+slope*maxX), class: "rp-user-line" }));
        const prediction = intercept+slope*attempts;
        lines.append(node("line", { x1: scaleX(attempts), x2: scaleX(attempts),
          y1: scaleY(0), y2: scaleY(prediction), class: "rp-guide" }));
        const marker = node("circle", { cx: scaleX(attempts), cy: scaleY(prediction),
          r: 5, class: "rp-marker" });
        marker.append(node("title", {}, `The selected line estimates ${format(prediction)} errors at ${attempts} attempts`));
        lines.append(marker);
      }

      function update() {
        const attempts = Number(control("attempts").value);
        const sse = squaredError(fitted, intercept, slope);
        const optimal = squaredError(fitted, fitted.intercept, fitted.slope);
        const extra = Math.max(0, 100*(sse/optimal-1));
        const atFit = Math.abs(intercept-fitted.intercept) < 1e-10 &&
          Math.abs(slope-fitted.slope) < 1e-10;
        value("line-label", atFit ? "Fitted line" : "Adjusted line");
        value("intercept", format(intercept, 5)); value("slope", format(slope, 5));
        value("attempts", String(attempts));
        value("equation", `Estimated errors = ${format(intercept, 5)} ${slope < 0 ? "−" : "+"} ${format(Math.abs(slope), 5)} × attempts`);
        value("r2", `${format(100*(1-sse/fitted.sst))}%`);
        value("rmse", format(Math.sqrt(sse/fitted.n), 3));
        const prediction = intercept+slope*attempts;
        value("prediction", format(prediction));
        const comparison = extra < 0.005 ?
          "At the least squares fit: no other straight line has lower squared error on these records." :
          `${format(extra)}% more squared error than the fitted line.`;
        value("feedback", comparison + (prediction < 0 ?
          " This line estimates a negative count here; that illustrates a limitation of linear regression." : ""));
        drawChart(attempts);
      }

      function restoreFit() {
        intercept = fitted.intercept; slope = fitted.slope;
        control("intercept").value = intercept; control("slope").value = slope;
        update();
      }

      function changeYear() {
        points = mergePoints(data.years, control("year").value);
        fitted = fit(points); plottedExtent = null;
        control("attempts").max = Math.max(...points.map(([x]) => x));
        value("records", `${format(fitted.n, 0)} player match records`);
        restoreFit();
      }

      control("year").addEventListener("change", changeYear);
      for (const name of ["intercept", "slope"]) {
        control(name).addEventListener("input", () => {
          if (name === "intercept") intercept = Number(control(name).value);
          else slope = Number(control(name).value);
          update();
        });
      }
      control("attempts").addEventListener("input", update);
      control("residuals").addEventListener("change", update);
      root.querySelector("[data-action=fit]").addEventListener("click", restoreFit);
      root.querySelector(".rp-interface").hidden = false;
      status.hidden = true;
      changeYear();
    } catch (error) {
      status.textContent = "The playground data could not load. The fitted coefficients and R chart above are still available; reload to try again.";
      console.error("Regression playground:", error);
    }
  }
  document.querySelectorAll("[data-regression]").forEach(initialize);
})();
