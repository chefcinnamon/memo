(() => {
  "use strict";

  const referencePrice = 3000;

  function deltaExample(price, gamma) {
    if (!Number.isFinite(price) || !Number.isFinite(gamma)) throw new RangeError("Finite inputs required");
    return { delta: 1 + gamma * (price - referencePrice), gamma, deltaChange: 10 * gamma };
  }

  // Reciprocal bounds keep the initial deposit at exactly 1 ETH + 3,000 USDC.
  function lpParameters(widthPercent) {
    if (!Number.isFinite(widthPercent) || widthPercent <= 0) throw new RangeError("Positive range width required");
    const factor = 1 + widthPercent / 100;
    const lower = referencePrice / factor;
    const upper = referencePrice * factor;
    const liquidity = 1 / (1 / Math.sqrt(referencePrice) - 1 / Math.sqrt(upper));
    return { lower, upper, liquidity };
  }

  function lpSnapshot(position, price) {
    if (!Number.isFinite(price) || price <= 0) throw new RangeError("Positive price required");
    const { lower, upper, liquidity } = position;
    const boundedPrice = Math.max(lower, Math.min(upper, price));
    const eth = Math.max(0, liquidity * (1 / Math.sqrt(boundedPrice) - 1 / Math.sqrt(upper)));
    const usdc = Math.max(0, liquidity * (Math.sqrt(boundedPrice) - Math.sqrt(lower)));
    const tolerance = 16 * Number.EPSILON * Math.max(price, upper);
    const atLower = Math.abs(price - lower) <= tolerance;
    const atUpper = Math.abs(price - upper) <= tolerance;
    const region = atLower ? "lower boundary" : atUpper ? "upper boundary" :
      price < lower ? "below range" : price > upper ? "above range" : "inside range";
    const gamma = atLower || atUpper ? null :
      price < lower || price > upper ? 0 : -liquidity / (2 * price ** 1.5);
    return { eth, usdc, value: price * eth + usdc, delta: eth, gamma, region,
      holdingValue: price + referencePrice };
  }

  // Equal opening value in both tokens: 0.5 ETH + 50 USDC at a price of 100.
  const payoffPosition = {
    lower: 80, upper: 125,
    liquidity: 50 / (Math.sqrt(100) - Math.sqrt(80))
  };

  function lpPayoffSnapshot(price) {
    if (!Number.isFinite(price) || price < 0) throw new RangeError("Nonnegative price required");
    const state = lpSnapshot(payoffPosition, Math.max(price, Number.MIN_VALUE));
    const initialValue = lpSnapshot(payoffPosition, 100).value;
    const value = price * state.eth + state.usdc;
    return { eth: state.eth, usdc: state.usdc, value, initialValue, profit: value - initialValue,
      delta: state.delta, gamma: state.gamma };
  }

  function lpHedgedSnapshot(price) {
    const lp = lpPayoffSnapshot(price);
    const borrowedEth = lpPayoffSnapshot(100).eth;
    const initialDebtValue = borrowedEth * 100;
    const saleProceeds = initialDebtValue;
    const debtValue = borrowedEth * price;
    const hedgeProfit = initialDebtValue - debtValue;
    const netEth = lp.eth - borrowedEth;
    const netUsdc = lp.usdc + saleProceeds;
    const netValue = lp.value + saleProceeds - debtValue;
    const netInitialValue = lp.initialValue + saleProceeds - initialDebtValue;
    return { ...lp, borrowedEth, debtValue, saleProceeds, hedgeProfit, netProfit: netValue - netInitialValue,
      hedgeDelta: -borrowedEth, hedgeGamma: 0, netEth, netUsdc,
      netDelta: netEth, netGamma: lp.gamma, netValue, netInitialValue };
  }

  // A Panoptic-style long put reverses the token exposure of its own LP range.
  // Match the two legs by their ETH notional below both ranges, not by liquidity L.
  function lpPutSpreadSnapshot(price, strike, widthPercent = 25) {
    if (!Number.isFinite(strike) || strike <= 0) throw new RangeError("Positive strike required");
    if (!Number.isFinite(widthPercent) || widthPercent <= 0) throw new RangeError("Positive range width required");
    const short = lpPayoffSnapshot(price);
    const notionalEth = lpPayoffSnapshot(0).eth;
    const factor = 1 + widthPercent / 100;
    const lower = strike / factor;
    const upper = strike * factor;
    const position = strike === 100 && widthPercent === 25 ? payoffPosition : {
      lower, upper, liquidity: notionalEth / (1 / Math.sqrt(lower) - 1 / Math.sqrt(upper))
    };
    const long = lpSnapshot(position, Math.max(price, Number.MIN_VALUE));
    const initialLongValue = lpSnapshot(position, 100).value;
    const longValue = price * long.eth + long.usdc;
    const netEth = short.eth - long.eth;
    const netUsdc = short.usdc - long.usdc;
    const netValue = price * netEth + netUsdc;
    const initialNetValue = short.initialValue - initialLongValue;
    return { strike, widthPercent, notionalEth, longRange: { lower, upper },
      shortProfit: short.profit, longProfit: initialLongValue - longValue,
      netProfit: netValue - initialNetValue, netValue, initialNetValue,
      netEth, netUsdc, netDelta: netEth,
      netGamma: short.gamma === null || long.gamma === null ? null : short.gamma - long.gamma };
  }

  // Original reference chart: both legs are normalized to one ETH of notional.
  // Both strikes are 100 and both ranges are 80–125.
  function lpShortStraddleSnapshot(price) {
    const shortPut = lpPayoffSnapshot(price);
    const notionalEth = lpPayoffSnapshot(0).eth;
    const shortCallProfit = shortPut.profit - notionalEth * (price - 100);
    return {
      price, notionalEth,
      shortPutProfit: shortPut.profit / notionalEth,
      shortCallProfit: shortCallProfit / notionalEth,
      netProfit: (shortPut.profit + shortCallProfit) / notionalEth,
      netEth: (2 * shortPut.eth - notionalEth) / notionalEth,
      netUsdc: 2 * shortPut.usdc / notionalEth
    };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { deltaExample, lpParameters, lpSnapshot, lpPayoffSnapshot, lpHedgedSnapshot, lpPutSpreadSnapshot, lpShortStraddleSnapshot };
  }
  if (typeof document === "undefined") return;

  const svgNS = "http://www.w3.org/2000/svg";
  const format = (number, digits = 2) => {
    const roundedZero = Math.abs(number) < 0.5 * 10 ** -digits ? 0 : number;
    return roundedZero.toLocaleString("en-US", {
      minimumFractionDigits: digits, maximumFractionDigits: digits
    }).replace("-", "−");
  };
  const signed = (number, digits = 4) => (number > 0 ? "+" : "") + format(number, digits);
  const node = (tag, attributes = {}, text) => {
    const result = document.createElementNS(svgNS, tag);
    Object.entries(attributes).forEach(([key, value]) => result.setAttribute(key, value));
    if (text !== undefined) result.textContent = text;
    return result;
  };
  const sample = (low, high, fn, count = 250) => Array.from({ length: count + 1 }, (_, i) => {
    const x = low + (high - low) * i / count;
    return [x, fn(x)];
  });

  function drawPlot(svg, options) {
    const width = Math.max(300, Math.min(700, Math.round(svg.getBoundingClientRect().width || 700)));
    const height = options.height || 190;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const frame = { left: 72, right: width - (options.rightAxis ? 54 : 14), top: 18, bottom: height - 46 };
    const sx = (x) => frame.left + (x - options.xMin) / (options.xMax - options.xMin) * (frame.right - frame.left);
    const sy = (y) => frame.bottom - (y - options.yMin) / (options.yMax - options.yMin) * (frame.bottom - frame.top);
    const syRight = (y) => frame.bottom - (y - options.rightAxis.min) /
      (options.rightAxis.max - options.rightAxis.min) * (frame.bottom - frame.top);
    let layer = svg.querySelector("[data-lp-layer]");
    if (!layer) { layer = node("g", { "data-lp-layer": "" }); svg.append(layer); }
    layer.replaceChildren();
    const clipID = `${svg.closest("[data-lp-playground]").id}-${svg.dataset.lpChart}-clip`;
    const defs = node("defs");
    const clip = node("clipPath", { id: clipID });
    clip.append(node("rect", { x: frame.left, y: frame.top,
      width: frame.right - frame.left, height: frame.bottom - frame.top }));
    defs.append(clip); layer.append(defs);

    const ranges = options.ranges || (options.range ? [options.range] : []);
    for (const range of ranges) {
      layer.append(node("rect", { x: sx(range.lower), y: frame.top,
        width: sx(range.upper) - sx(range.lower),
        height: frame.bottom - frame.top, class: range.className || "lp-band" }));
      for (const bound of [range.lower, range.upper]) {
        layer.append(node("line", { x1: sx(bound), x2: sx(bound), y1: frame.top,
          y2: frame.bottom, class: "lp-bound" }));
      }
    }
    const xTicks = width < 480 ? options.mobileXTicks : options.xTicks;
    for (const x of xTicks) {
      layer.append(node("line", { x1: sx(x), x2: sx(x), y1: frame.top,
        y2: frame.bottom, class: "lp-grid" }));
      const anchor = x === options.xMin ? "start" : x === options.xMax ? "end" : "middle";
      layer.append(node("text", { x: sx(x), y: frame.bottom + 20,
        "text-anchor": anchor, class: "lp-axis" }, format(x, 0)));
    }
    for (const y of options.yTicks) {
      layer.append(node("line", { x1: frame.left, x2: frame.right, y1: sy(y),
        y2: sy(y), class: y === 0 ? "lp-zero" : "lp-grid" }));
      layer.append(node("text", { x: frame.left - 8, y: sy(y) + 4,
        "text-anchor": "end", class: options.axisClass || "lp-axis" }, format(y, options.yDigits || 0)));
    }
    if (options.rightAxis) {
      for (const y of options.rightAxis.ticks) {
        layer.append(node("text", { x: frame.right + 8, y: syRight(y) + 4,
          "text-anchor": "start", class: options.rightAxis.className || "lp-axis" },
        format(y, options.rightAxis.digits || 0)));
      }
      layer.append(node("text", { x: frame.right, y: 12, "text-anchor": "end",
        class: options.rightAxis.className || "lp-axis" }, options.rightAxis.unit));
    }
    layer.append(node("text", { x: (frame.left + frame.right) / 2, y: height - 5,
      "text-anchor": "middle", class: "lp-axis-label" }, options.xLabel || "ETH price · USDC per ETH"));
    layer.append(node("text", { x: frame.left, y: 12, class: options.axisClass || "lp-axis" }, options.unit));

    const curves = node("g", { "clip-path": `url(#${clipID})` });
    for (const curve of options.curves) {
      const scaleY = curve.axis === "right" ? syRight : sy;
      const path = curve.points.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(2)},${scaleY(y).toFixed(2)}`).join(" ");
      curves.append(node("path", { d: path, class: curve.className || "lp-curve" }));
    }
    layer.append(curves);
    for (const range of ranges) {
      if (range.label) layer.append(node("text", {
        x: sx((range.lower + range.upper) / 2),
        y: range.labelPosition === "bottom" ? frame.bottom - 10 : frame.top + 15,
        "text-anchor": "middle", class: "lp-range-label " + (range.labelClassName || "")
      }, range.label));
    }
    for (const label of options.curveLabels || []) {
      const x = sx(label.x);
      const y = sy(label.y);
      const text = node("text", { x: frame.right - 8, y: y + label.dy,
        "text-anchor": "end", class: "lp-curve-label " + label.className }, label.text);
      layer.append(text);
      const box = text.getBBox();
      layer.insertBefore(node("line", { x1: x, y1: y,
        x2: box.x + box.width / 2, y2: label.dy > 0 ? box.y - 3 : box.y + box.height + 3,
        class: "lp-curve-callout " + label.calloutClassName }), text);
      layer.insertBefore(node("circle", { cx: x, cy: y, r: 3,
        class: "lp-marker " + label.markerClassName }), text);
    }
    if (Number.isFinite(options.price)) {
      layer.append(node("line", { x1: sx(options.price), x2: sx(options.price),
        y1: frame.top, y2: frame.bottom, class: "lp-price-guide" }));
    }
    for (const marker of options.markers || []) {
      const dot = node("circle", { cx: sx(marker.x), cy: sy(marker.y), r: marker.open ? 3.5 : 5,
        class: marker.className || (marker.open ? "lp-open-marker" : "lp-marker") });
      dot.append(node("title", {}, marker.title)); layer.append(dot);
    }
  }

  function initialize(root) {
    if (root.dataset.lpInitialized) return;
    root.dataset.lpInitialized = "true";
    const control = (name) => root.querySelector(`[data-lp-control="${name}"]`);
    const output = (name, text) => { root.querySelectorAll(`[data-lp-output="${name}"]`).forEach((item) => { item.textContent = text; }); };
    const value = (name, text) => { root.querySelector(`[data-lp-value="${name}"]`).textContent = text; };
    const chart = (name) => root.querySelector(`[data-lp-chart="${name}"]`);
    const feedback = root.querySelector("[data-lp-feedback]");
    const resets = root.querySelectorAll("[data-lp-reset]");
    const isSigns = root.dataset.lpPlayground === "signs";
    const isPayoff = root.dataset.lpPlayground === "payoff";
    const isHedge = root.dataset.lpPlayground === "hedge";
    const isPutSpread = ["put-spread", "put-range"].includes(root.dataset.lpPlayground);
    const isPutRange = root.dataset.lpPlayground === "put-range";
    const isShortStraddle = root.dataset.lpPlayground === "short-straddle";

    function drawSigns() {
      const price = Number(control("price").value);
      const gamma = Number(control("gamma").value);
      const state = deltaExample(price, gamma);
      output("price", format(price, 0)); output("gamma", signed(gamma));
      control("price").setAttribute("aria-valuetext", `${price} USDC per ETH`);
      control("gamma").setAttribute("aria-valuetext", `${gamma} change in delta per 1 USDC price increase`);
      value("delta", format(state.delta, 4)); value("gamma", signed(gamma));
      value("delta-change", signed(state.deltaChange));
      root.querySelectorAll("[data-lp-preset]").forEach((button) => {
        button.setAttribute("aria-pressed", String(Number(button.dataset.lpPreset) === gamma));
      });
      drawPlot(chart("signs"), { height: 250, xMin: 2500, xMax: 3500, yMin: -0.1, yMax: 2.1,
        xTicks: [2500, 2750, 3000, 3250, 3500], mobileXTicks: [2500, 3000, 3500],
        yTicks: [0, 0.5, 1, 1.5, 2], yDigits: 1, unit: "Delta", price,
        curves: [{ points: sample(2500, 3500, (p) => deltaExample(p, gamma).delta) }],
        markers: [{ x: price, y: state.delta, title: `At ${format(price, 0)} USDC per ETH, delta is ${format(state.delta, 4)}` }] });
      feedback.textContent = gamma > 0 ?
        "Positive gamma: delta increases when price rises, and decreases when price falls." : gamma < 0 ?
        "Negative gamma: delta decreases when price rises, and increases when price falls." :
        "Zero gamma: price changes leave delta constant.";
    }

    function drawPosition() {
      const price = Number(control("price").value);
      const position = lpParameters(Number(control("width").value));
      const state = lpSnapshot(position, price);
      const { lower, upper, liquidity } = position;
      output("price", format(price, 0)); output("range", `${format(lower, 0)} to ${format(upper, 0)}`);
      control("price").setAttribute("aria-valuetext", `${price} USDC per ETH`);
      control("width").setAttribute("aria-valuetext", `Range ${format(lower, 0)} to ${format(upper, 0)} USDC per ETH`);
      value("value", format(state.value)); value("delta", format(state.delta, 4));
      value("gamma", state.gamma === null ? "Undefined" : format(state.gamma, 6));
      root.querySelector('[data-lp-value="gamma"]').title = state.gamma === null ?
        "Undefined at the range boundary because delta has different slopes on either side." : "";
      root.querySelector("[data-lp-balances]").textContent =
        `Currently held: ${format(state.eth, 4)} ETH + ${format(state.usdc)} USDC.`;

      // Include bounds and both marked prices in continuous value/delta curves.
      const prices = [...new Set([...sample(1000, 6500, () => 0).map(([p]) => p), lower, upper, referencePrice, price])].sort((a, b) => a - b);
      const snapshots = prices.map((p) => [p, lpSnapshot(position, p)]);
      const common = { xMin: 1000, xMax: 6500, xTicks: [1000, 2000, 3000, 4000, 5000, 6000],
        mobileXTicks: [1000, 3000, 5000, 6500], price, range: position };
      drawPlot(chart("value"), { ...common, yMin: 0, yMax: 10000, yTicks: [0, 5000, 10000], unit: "USDC",
        curves: [{ points: [[1000, 4000], [6500, 9500]], className: "lp-hold" },
          { points: snapshots.map(([p, s]) => [p, s.value]) }],
        markers: [{ x: price, y: state.value, title: `LP value: ${format(state.value)} USDC` }] });
      drawPlot(chart("delta"), { ...common, yMin: -0.1, yMax: 2.7, yTicks: [0, 1, 2], unit: "ETH",
        curves: [{ points: snapshots.map(([p, s]) => [p, s.delta]) }],
        markers: [{ x: price, y: state.delta, title: `Delta: ${format(state.delta, 4)} ETH` }] });

      const insideGamma = (p) => -liquidity / (2 * p ** 1.5);
      const lowGamma = insideGamma(lower);
      // Separate paths and open endpoints preserve the discontinuities in gamma.
      const gammaCurves = [
        { points: [[1000, 0], [lower, 0]] },
        { points: sample(lower, upper, insideGamma) },
        { points: [[upper, 0], [6500, 0]] }
      ];
      const gammaMarkers = [lower, upper].flatMap((bound) => [0, insideGamma(bound)].map((g) =>
        ({ x: bound, y: g, open: true, title: "Gamma is undefined at the exact boundary; this circle marks a one-sided limit." })));
      if (state.gamma !== null) gammaMarkers.push({ x: price, y: state.gamma, title: `Gamma: ${format(state.gamma, 6)}` });
      drawPlot(chart("gamma"), { ...common, yMin: lowGamma * 1.2, yMax: -lowGamma * 0.15,
        yTicks: [lowGamma, lowGamma / 2, 0], yDigits: 5, unit: "Gamma", curves: gammaCurves, markers: gammaMarkers });

      feedback.textContent = state.region === "inside range" ?
        "Inside the range: negative gamma.\nETH price ↑, ETH held ↓, delta ↓\nETH price ↓, ETH held ↑, delta ↑" :
        state.region === "below range" ?
        "Below the range: only ETH is held. Delta is constant and gamma is zero. The position still changes in value when ETH prices change." :
        state.region === "above range" ?
        "Above the range: only USDC is held. Delta and gamma are zero." :
        `At the ${state.region}: gamma switches between zero and negative. It has no single value at the exact boundary.`;
    }

    function drawPayoff() {
      const price = Number(control("price").value);
      const state = lpPayoffSnapshot(price);
      output("price", format(price, 0));
      root.querySelectorAll('[data-lp-control="price"]').forEach((input) =>
        input.setAttribute("aria-valuetext", price + " USDC per ETH"));
      value("profit", signed(state.profit, 2));
      value("eth", format(state.eth, 4));
      value("usdc", format(state.usdc));
      const prices = [...new Set([...sample(0, 200, () => 0, 400).map(([p]) => p),
        payoffPosition.lower, payoffPosition.upper, 100, price])].sort((a, b) => a - b);
      const snapshots = prices.map((p) => [p, lpPayoffSnapshot(p)]);
      const common = { xMin: 0, xMax: 200, price, range: payoffPosition,
        xLabel: "ETH price · USDC",
        xTicks: [0, 25, 50, 75, 100, 125, 150, 175, 200],
        mobileXTicks: [0, 50, 100, 150, 200] };
      drawPlot(chart("profit"), { ...common, height: 350, yMin: -50, yMax: 50,
        yTicks: [-50, -25, 0, 25, 50], unit: "Profit / loss · USDC",
        curves: [{ points: snapshots.map(([p, s]) => [p, s.profit]),
          className: "lp-curve lp-payoff-curve" }] });
      drawPlot(chart("balances"), { ...common, height: 250, yMin: 0, yMax: 1.1,
        yTicks: [0, 0.25, 0.5, 0.75, 1], yDigits: 2, unit: "ETH held",
        axisClass: "lp-axis lp-axis-eth",
        rightAxis: { min: 0, max: 110, ticks: [0, 25, 50, 75, 100],
          unit: "USDC held", className: "lp-axis lp-axis-usdc" },
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.eth]), className: "lp-curve lp-eth-curve" },
          { points: snapshots.map(([p, s]) => [p, s.usdc]), axis: "right",
            className: "lp-curve lp-usdc-curve" }
        ] });
      feedback.textContent = price <= 80 ?
        "At 80 and below: " + format(state.eth, 4) + " ETH and no USDC. Profit changes with ETH's price." : price >= 125 ?
        "At 125 and above: no ETH and " + format(state.usdc) + " USDC. Profit stays at " + signed(state.profit, 2) + " USDC, excluding fees." :
        "Within the range: rising prices reduce ETH held and increase USDC held.";
    }

    function drawHedge() {
      const prices = [...new Set([...sample(0, 200, () => 0, 400).map(([p]) => p),
        payoffPosition.lower, payoffPosition.upper, 100])].sort((a, b) => a - b);
      const snapshots = prices.map((p) => [p, lpHedgedSnapshot(p)]);
      const common = { xMin: 0, xMax: 200, range: payoffPosition,
        xLabel: "ETH price · USDC", height: 350,
        xTicks: [0, 25, 50, 75, 100, 125, 150, 175, 200],
        mobileXTicks: [0, 50, 100, 150, 200] };
      drawPlot(chart("hedge-profit"), { ...common, yMin: -30, yMax: 30,
        yTicks: [-30, -20, -10, 0, 10, 20, 30], unit: "Profit / loss · USDC",
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.profit]), className: "lp-curve lp-payoff-curve" },
          { points: snapshots.map(([p, s]) => [p, s.hedgeProfit]), className: "lp-curve lp-short-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netProfit]), className: "lp-curve lp-net-curve" }
        ] });
      drawPlot(chart("hedge-balances"), { ...common, yMin: -1, yMax: 1,
        yTicks: [-1, -0.5, 0, 0.5, 1], yDigits: 2, unit: "Net ETH",
        axisClass: "lp-axis lp-axis-eth",
        rightAxis: { min: 0, max: 200, ticks: [0, 50, 100, 150, 200],
          unit: "USDC", className: "lp-axis lp-axis-usdc" },
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.netEth]), className: "lp-curve lp-eth-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netUsdc]), axis: "right",
            className: "lp-curve lp-usdc-curve" }
        ] });
      const opening = lpHedgedSnapshot(100);
      drawPlot(chart("hedge-delta"), { ...common, height: 300, yMin: -0.6, yMax: 1.15,
        yTicks: [-0.5, 0, 0.5, 1], yDigits: 2, unit: "Delta · ETH", price: 100,
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.delta]), className: "lp-curve lp-payoff-curve" },
          { points: [[0, opening.hedgeDelta], [200, opening.hedgeDelta]], className: "lp-curve lp-short-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netDelta]), className: "lp-curve lp-net-curve" }
        ],
        markers: [
          { x: 100, y: opening.delta, className: "lp-marker lp-marker-profit", title: "Opening LP delta: +0.5 ETH" },
          { x: 100, y: opening.hedgeDelta, className: "lp-marker lp-marker-short", title: "ETH debt delta: −0.5 ETH" },
          { x: 100, y: opening.netDelta, className: "lp-marker lp-marker-net", title: "Opening net delta: 0 ETH" }
        ] });
      const insideGamma = (p) => -payoffPosition.liquidity / (2 * p ** 1.5);
      const gammaPaths = [
        [[0, 0], [payoffPosition.lower, 0]],
        sample(payoffPosition.lower, payoffPosition.upper, insideGamma),
        [[payoffPosition.upper, 0], [200, 0]]
      ];
      const gammaMarkers = [payoffPosition.lower, payoffPosition.upper].flatMap((bound) =>
        [0, insideGamma(bound)].map((g) => ({ x: bound, y: g, open: true,
          title: "LP and net gamma are undefined at this boundary; the circles mark their limits." })));
      drawPlot(chart("hedge-gamma"), { ...common, height: 300, yMin: -0.04, yMax: 0.006,
        yTicks: [-0.04, -0.02, 0], yDigits: 3, unit: "Gamma",
        curves: [
          ...gammaPaths.map((points) => ({ points, className: "lp-curve lp-payoff-curve lp-gamma-lp" })),
          { points: [[0, 0], [200, 0]], className: "lp-curve lp-short-curve" },
          ...gammaPaths.map((points) => ({ points, className: "lp-curve lp-net-curve lp-gamma-net" }))
        ], markers: gammaMarkers });
    }

    function drawPutSpread() {
      const strike = Number(root.dataset.lpStrike);
      const widthPercent = Number(root.dataset.lpPutWidth || 25);
      const factor = 1 + widthPercent / 100;
      const longRange = { lower: strike / factor, upper: strike * factor };
      const prices = [...new Set([...sample(0, 200, () => 0, 400).map(([p]) => p),
        payoffPosition.lower, payoffPosition.upper, longRange.lower, longRange.upper, 100])]
        .sort((a, b) => a - b);
      const snapshots = prices.map((p) => [p, lpPutSpreadSnapshot(p, strike, widthPercent)]);
      const common = { xMin: 0, xMax: 200, height: 350, xLabel: "ETH price · USDC",
        xTicks: [0, 25, 50, 75, 100, 125, 150, 175, 200], mobileXTicks: [0, 50, 100, 150, 200],
        ranges: [
          { ...payoffPosition, label: "SELL PUT RANGE", labelClassName: "lp-label-sell", labelPosition: isPutRange ? "bottom" : "top" },
          { ...longRange, className: "lp-long-band", label: "BUY PUT RANGE", labelClassName: "lp-label-buy", labelPosition: isPutRange ? "top" : "bottom" }
        ] };
      const sellLabel = lpPutSpreadSnapshot(160, strike, widthPercent);
      const buyLabelPrice = isPutRange && widthPercent > 25 ? 120 : 160;
      const buyLabel = lpPutSpreadSnapshot(buyLabelPrice, strike, widthPercent);
      drawPlot(chart("spread-profit"), { ...common, yMin: isPutRange ? -15 : -50, yMax: isPutRange ? 10 : 50,
        yTicks: isPutRange ? [-15, -10, -5, 0, 5, 10] : [-50, -25, 0, 25, 50], unit: "Profit / loss · USDC",
        ranges: common.ranges.map(({ label, ...range }) => range),
        curveLabels: [
          { x: 160, y: sellLabel.shortProfit, dy: -14, text: "SELL PUT (LP)",
            className: "lp-label-sell", calloutClassName: "lp-callout-sell", markerClassName: "lp-marker-profit" },
          { x: buyLabelPrice, y: buyLabel.longProfit, dy: 20, text: "BUY PUT",
            className: "lp-label-buy", calloutClassName: "lp-callout-buy", markerClassName: "lp-marker-short" }
        ],
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.shortProfit]), className: "lp-curve lp-payoff-curve" },
          { points: snapshots.map(([p, s]) => [p, s.longProfit]), className: "lp-curve lp-short-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netProfit]), className: "lp-curve lp-net-curve" }
        ] });
      drawPlot(chart("spread-balances"), { ...common, height: 300, yMin: -1, yMax: 1,
        yTicks: [-1, -0.5, 0, 0.5, 1], yDigits: 2, unit: "Net ETH",
        axisClass: "lp-axis lp-axis-eth",
        rightAxis: { min: isPutRange ? -100 : -180, max: isPutRange ? 100 : 180,
          ticks: isPutRange ? [-100, -50, 0, 50, 100] : [-180, -90, 0, 90, 180],
          unit: "Net USDC", className: "lp-axis lp-axis-usdc" },
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.netEth]), className: "lp-curve lp-eth-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netUsdc]), axis: "right", className: "lp-curve lp-usdc-curve" }
        ] });
    }

    function drawShortStraddle() {
      const prices = [...new Set([...sample(0, 200, () => 0, 400).map(([p]) => p),
        payoffPosition.lower, payoffPosition.upper, 100])]
        .sort((a, b) => a - b);
      const snapshots = prices.map((p) => [p, lpShortStraddleSnapshot(p)]);
      const common = { xMin: 0, xMax: 200, range: payoffPosition, xLabel: "ETH price · USDC",
        xTicks: [0, 25, 50, 75, 100, 125, 150, 175, 200],
        mobileXTicks: [0, 50, 100, 150, 200] };
      drawPlot(chart("straddle-profit"), { ...common, height: 350, yMin: -50, yMax: 10,
        yTicks: [-50, -40, -30, -20, -10, 0, 10], unit: "Profit / loss · USDC",
        curveLabels: [
          { x: 160, y: lpShortStraddleSnapshot(160).shortPutProfit, dy: -14, text: "SELL PUT (LP)",
            className: "lp-label-sell", calloutClassName: "lp-callout-sell", markerClassName: "lp-marker-profit" },
          { x: 120, y: lpShortStraddleSnapshot(120).shortCallProfit, dy: 20, text: "SELL CALL",
            className: "lp-label-buy", calloutClassName: "lp-callout-buy", markerClassName: "lp-marker-short" }
        ],
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.shortPutProfit]), className: "lp-curve lp-payoff-curve" },
          { points: snapshots.map(([p, s]) => [p, s.shortCallProfit]), className: "lp-curve lp-short-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netProfit]), className: "lp-curve lp-net-curve" }
        ] });
      drawPlot(chart("straddle-balances"), { ...common, height: 300, yMin: -1, yMax: 1,
        yTicks: [-1, -0.5, 0, 0.5, 1], yDigits: 2, unit: "Net ETH",
        axisClass: "lp-axis lp-axis-eth",
        rightAxis: { min: -250, max: 250, ticks: [-250, -125, 0, 125, 250],
          unit: "Net USDC", className: "lp-axis lp-axis-usdc" },
        curves: [
          { points: snapshots.map(([p, s]) => [p, s.netEth]), className: "lp-curve lp-eth-curve" },
          { points: snapshots.map(([p, s]) => [p, s.netUsdc]), axis: "right", className: "lp-curve lp-usdc-curve" }
        ] });
    }

    const render = isSigns ? drawSigns : isPayoff ? drawPayoff : isHedge ? drawHedge :
      isShortStraddle ? drawShortStraddle : isPutSpread ? drawPutSpread : drawPosition;
    try {
      root.querySelectorAll("[data-lp-control]").forEach((input) => input.addEventListener("input", () => {
        if (isPayoff && input.dataset.lpControl === "price") {
          root.querySelectorAll('[data-lp-control="price"]').forEach((peer) => { peer.value = input.value; });
        }
        render();
      }));
      root.querySelectorAll("[data-lp-preset]").forEach((button) => button.addEventListener("click", () => {
        control("gamma").value = button.dataset.lpPreset; render();
      }));
      resets.forEach((button) => button.addEventListener("click", () => {
        root.querySelectorAll('[data-lp-control="price"]').forEach((input) => { input.value = isPayoff ? "100" : "3000"; });
        if (!isPayoff) control(isSigns ? "gamma" : "width").value = isSigns ? "-0.001" : "50";
        render();
      }));
      root.querySelector("[data-lp-interface]").hidden = false;
      render();
      root.querySelector("[data-lp-loading]").hidden = true;
      resets.forEach((button) => { button.hidden = false; });
      let lastWidth = root.getBoundingClientRect().width;
      if (typeof ResizeObserver !== "undefined") {
        new ResizeObserver(() => {
          const nextWidth = root.getBoundingClientRect().width;
          if (Math.abs(nextWidth - lastWidth) > 1) { lastWidth = nextWidth; render(); }
        }).observe(root);
      }
    } catch (error) {
      root.querySelector("[data-lp-interface]").hidden = true;
      root.querySelector("[data-lp-loading]").hidden = false;
      root.querySelector("[data-lp-loading]").textContent = "The playground could not load. The definitions and formulas above remain available.";
      console.error("LP playground:", error);
    }
  }

  document.querySelectorAll(".lp-strategy-title").forEach((heading) => {
    const symbols = "!@#$%&*=?/\\|<[]{}0123456789";
    const colors = ["var(--ascii-purple)", "var(--ascii-blue)", "var(--ascii-pink)", "var(--ascii-teal)"];
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const glyphs = [];
    heading.querySelectorAll(".lp-strategy-ascii").forEach((decoration) => {
      const right = decoration.classList.contains("lp-strategy-ascii-right");
      const count = right ? 7 : 9;
      decoration.replaceChildren(...Array.from({ length: count }, (_, index) => {
        const glyph = document.createElement("span");
        glyph.textContent = "@";
        glyphs.push({ glyph, fixed: right ? index === 0 : index === count - 1 });
        return glyph;
      }));
    });
    const shuffle = () => glyphs.forEach(({ glyph, fixed }) => {
      glyph.textContent = fixed ? "@" : symbols[Math.floor(Math.random() * symbols.length)];
      glyph.style.color = colors[Math.floor(Math.random() * colors.length)];
    });
    shuffle();
    window.setInterval(() => {
      if (document.hidden || reducedMotion.matches) return;
      const bounds = heading.getBoundingClientRect();
      if (bounds.bottom > 0 && bounds.top < window.innerHeight) shuffle();
    }, 520);
  });

  document.querySelectorAll("[data-lp-playground]").forEach(initialize);
})();
