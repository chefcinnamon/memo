const test = require('node:test');
const assert = require('node:assert/strict');
const { lpParameters, lpSnapshot, lpPayoffSnapshot, lpHedgedSnapshot, lpPutSpreadSnapshot } = require('../assets/js/lp-playground.js');

function close(actual, expected, tolerance = 1e-7) {
  assert.ok(Math.abs(actual - expected) <= tolerance,
    `Expected ${actual} to agree with ${expected} within ${tolerance}`);
}

test('Every selectable range preserves the post’s original deposit', () => {
  for (let width = 10; width <= 100; width += 5) {
    const initial = lpSnapshot(lpParameters(width), 3000);
    close(initial.eth, 1);
    close(initial.usdc, 3000);
    close(initial.value, 6000);
    close(initial.delta, 1);
  }
});

test('Numerical derivatives independently verify delta and gamma in all regions', () => {
  for (const width of [10, 25, 50, 100]) {
    const position = lpParameters(width);
    const prices = [position.lower / 2, position.lower * 0.9,
      position.lower * 1.01, 3000, position.upper * 0.99, position.upper * 1.1];
    for (const price of prices) {
      const state = lpSnapshot(position, price);
      const step = 0.1;
      const below = lpSnapshot(position, price - step);
      const above = lpSnapshot(position, price + step);
      close((above.value - below.value) / (2 * step), state.delta);
      close((above.delta - below.delta) / (2 * step), state.gamma, 1e-8);
      close((above.value - 2 * state.value + below.value) / step ** 2, state.gamma);
    }
  }
});

test('Out of range, only one token remains and gamma is zero', () => {
  const position = lpParameters(50);
  const low = lpSnapshot(position, 1000);
  const lower = lpSnapshot(position, 1500);
  const high = lpSnapshot(position, 5000);
  const higher = lpSnapshot(position, 6500);
  close(low.usdc, 0);
  close(low.eth, lower.eth);
  close(low.gamma, 0);
  assert.ok(low.delta > 0);
  close(high.eth, 0);
  close(high.usdc, higher.usdc);
  close(high.value, higher.value);
  close(high.delta, 0);
  close(high.gamma, 0);
});

test('Value and delta remain continuous at bounds; gamma has different one-sided limits', () => {
  const position = lpParameters(50);
  for (const bound of [position.lower, position.upper]) {
    const at = lpSnapshot(position, bound);
    const before = lpSnapshot(position, bound - 0.0001);
    const after = lpSnapshot(position, bound + 0.0001);
    assert.equal(at.gamma, null);
    close(before.delta, at.delta, 1e-6);
    close(after.delta, at.delta, 1e-6);
    close(before.value, at.value, 0.001);
    close(after.value, at.value, 0.001);
    assert.ok((before.gamma === 0 && after.gamma < 0) || (before.gamma < 0 && after.gamma === 0));
  }
});

test('Fee-free LP value never exceeds holding the same initial tokens', () => {
  for (const width of [10, 50, 100]) {
    const position = lpParameters(width);
    for (let price = 1000; price <= 6500; price += 10) {
      const state = lpSnapshot(position, price);
      assert.ok(state.value <= state.holdingValue + 1e-8);
      assert.ok(state.eth >= 0 && state.usdc >= 0);
      assert.ok(state.gamma === null || state.gamma <= 0);
    }
  }
});

test('Widening the range reduces gamma magnitude at the same initial capital and delta', () => {
  let previous = Infinity;
  for (let width = 10; width <= 100; width += 5) {
    const state = lpSnapshot(lpParameters(width), 3000);
    assert.ok(Math.abs(state.gamma) < previous);
    previous = Math.abs(state.gamma);
  }
});

test('Undefined financial inputs are rejected', () => {
  for (const width of [0, -1, NaN, Infinity]) assert.throws(() => lpParameters(width), RangeError);
  for (const price of [0, -1, NaN, Infinity]) assert.throws(() => lpSnapshot(lpParameters(50), price), RangeError);
});

test('The 50/50 example preserves its opening deposit and both endpoint balances', () => {
  const initial = lpPayoffSnapshot(100);
  close(initial.eth, 0.5);
  close(initial.usdc, 50);
  close(initial.initialValue, 100);
  close(initial.profit, 0);
  for (const price of [0, 50, 80]) {
    const state = lpPayoffSnapshot(price);
    close(state.eth, 1.0590169943749474);
    close(state.usdc, 0);
    close(state.value, price * 1.0590169943749474);
    close(state.profit, price * 1.0590169943749474 - initial.value);
  }
  for (const price of [125, 150, 200]) {
    const state = lpPayoffSnapshot(price);
    close(state.eth, 0);
    close(state.usdc, 105.90169943749474);
    close(state.value, 105.90169943749474);
    close(state.profit, 5.90169943749474);
  }
  for (const price of [-1, NaN, Infinity]) assert.throws(() => lpPayoffSnapshot(price), RangeError);
});

test('The profit curve has ETH held as its slope and bends down inside the range', () => {
  for (const price of [25, 79, 81, 100, 124, 126, 175]) {
    const step = 0.001;
    const before = lpPayoffSnapshot(price - step);
    const at = lpPayoffSnapshot(price);
    const after = lpPayoffSnapshot(price + step);
    close((after.profit - before.profit) / (2 * step), at.eth);
    const slopeChange = (after.eth - before.eth) / (2 * step);
    if (price > 80 && price < 125) assert.ok(slopeChange < 0);
    else close(slopeChange, 0);
  }
});

test('The hedge keeps ETH sale proceeds as cash and deducts the fixed ETH debt', () => {
  const opening = lpHedgedSnapshot(100);
  close(opening.borrowedEth, 0.5);
  close(opening.netEth, 0);
  close(opening.saleProceeds, 50);
  close(opening.netUsdc, 100);
  close(opening.netValue, 100);
  close(opening.netInitialValue, 100);
  close(opening.debtValue, 50);
  close(opening.netProfit, 0);
  close(opening.hedgeDelta, -0.5);
  close(opening.hedgeGamma, 0);
  close(opening.gamma, -0.02368033988749895);
  close(opening.netGamma, opening.gamma);
  for (let price = 0; price <= 200; price += 0.5) {
    const state = lpHedgedSnapshot(price);
    close(state.netValue - opening.netValue, state.netProfit);
    close(state.netEth + state.borrowedEth, state.eth);
    close(state.netUsdc, state.usdc + 50);
    close(state.netValue, state.eth * price + state.usdc + 50 - state.debtValue);
    close(state.netProfit, state.profit + state.hedgeProfit);
    assert.ok(state.netProfit <= 1e-8, 'Opening tangent bounds the concave LP value');
  }
  close(lpHedgedSnapshot(80).netEth, 0.5590169943749474);
  close(lpHedgedSnapshot(125).netEth, -0.5);
  close(lpHedgedSnapshot(125).netUsdc, 155.90169943749474);
  close(lpHedgedSnapshot(0).netUsdc, 50);
  close(lpHedgedSnapshot(80).hedgeProfit, 10);
  close(lpHedgedSnapshot(125).hedgeProfit, -12.5);
  for (const bound of [80, 125]) {
    assert.equal(lpHedgedSnapshot(bound).netGamma, null);
  }
});

test('Hedged P&L derivatives verify that ETH debt removes opening delta but leaves gamma', () => {
  for (const price of [25, 79, 81, 100, 124, 126, 175]) {
    const step = 0.001;
    const before = lpHedgedSnapshot(price - step);
    const at = lpHedgedSnapshot(price);
    const after = lpHedgedSnapshot(price + step);
    close((after.netProfit - before.netProfit) / (2 * step), at.netDelta);
    const hedgedGamma = (after.netDelta - before.netDelta) / (2 * step);
    close(hedgedGamma, at.netGamma);
    close(hedgedGamma, (after.eth - before.eth) / (2 * step));
    if (price > 80 && price < 125) assert.ok(hedgedGamma < 0);
    else close(hedgedGamma, 0);
  }
});

test('Lower put strike caps the loss and higher strike produces a bearish spread', () => {
  const low = lpPutSpreadSnapshot(0, 80);
  close(low.netEth, 0);
  close(low.netUsdc, 0);
  close(low.netProfit, -15.278640450004204);
  close(lpPutSpreadSnapshot(64, 80).netProfit, low.netProfit);
  close(lpPutSpreadSnapshot(200, 80).netProfit, 5.90169943749474);
  close(lpPutSpreadSnapshot(200, 80).netUsdc, 21.180339887498945);
  close(lpPutSpreadSnapshot(0, 125).netProfit, 5.90169943749474);
  close(lpPutSpreadSnapshot(200, 125).netProfit, -20.573725421878945);
  close(lpPutSpreadSnapshot(200, 125).netUsdc, -26.475424859373685);
  for (const strike of [80, 125]) {
    let previous = lpPutSpreadSnapshot(0, strike).netProfit;
    for (let price = 0.5; price <= 200; price += 0.5) {
      const state = lpPutSpreadSnapshot(price, strike);
      assert.ok(strike < 100 ? state.netProfit >= previous - 1e-8 : state.netProfit <= previous + 1e-8);
      close(state.netProfit, state.shortProfit + state.longProfit);
      close(state.netValue, price * state.netEth + state.netUsdc);
      close(state.netProfit, state.netValue - state.initialNetValue);
      previous = state.netProfit;
    }
  }
});

test('Matching perpetual put range and notional cancel both token exposures at every price', () => {
  for (let price = 0; price <= 200; price += 0.5) {
    const state = lpPutSpreadSnapshot(price, 100);
    close(state.netEth, 0);
    close(state.netUsdc, 0);
    close(state.netProfit, 0);
    close(state.shortProfit, -state.longProfit);
  }
});

test('Put spread value is continuous at every boundary and its slope equals net ETH exposure', () => {
  for (const strike of [80, 100, 125]) {
    const { longRange } = lpPutSpreadSnapshot(100, strike);
    for (const bound of [80, 125, longRange.lower, longRange.upper]) {
      const before = lpPutSpreadSnapshot(bound - 0.000001, strike);
      const after = lpPutSpreadSnapshot(bound + 0.000001, strike);
      close(before.netProfit, after.netProfit, 0.00001);
      close(before.netEth, after.netEth, 0.000001);
      close(before.netUsdc, after.netUsdc, 0.0001);
    }
    for (const price of [30, 70, 90, 110, 140, 175]) {
      const step = 0.001;
      const before = lpPutSpreadSnapshot(price - step, strike);
      const at = lpPutSpreadSnapshot(price, strike);
      const after = lpPutSpreadSnapshot(price + step, strike);
      close((after.netProfit - before.netProfit) / (2 * step), at.netEth);
      close((after.netEth - before.netEth) / (2 * step), at.netGamma);
    }
    close(lpPutSpreadSnapshot(100, strike).netProfit, 0);
  }
  for (const strike of [0, -1, NaN, Infinity]) assert.throws(() => lpPutSpreadSnapshot(100, strike), RangeError);
  for (const price of [-1, NaN, Infinity]) assert.throws(() => lpPutSpreadSnapshot(price, 80), RangeError);
});

test('Wider and narrower buy put ranges reproduce the hump and dip with the same strike', () => {
  const wide = lpPutSpreadSnapshot(100, 100, 100);
  assert.deepEqual(wide.longRange, { lower: 50, upper: 200 });
  close(wide.netProfit, 0);
  close(lpPutSpreadSnapshot(0, 100, 100).netProfit, -12.26815962926102);
  close(lpPutSpreadSnapshot(200, 100, 100).netProfit, -12.26815962926102);
  assert.ok(lpPutSpreadSnapshot(103.9, 100, 100).netProfit > 0);
  const narrow = lpPutSpreadSnapshot(100, 100, 5);
  close(narrow.longRange.lower, 95.23809523809524);
  close(narrow.longRange.upper, 105);
  close(narrow.netProfit, 0);
  close(lpPutSpreadSnapshot(0, 100, 5).netProfit, 4.61002317005007);
  close(lpPutSpreadSnapshot(200, 100, 5).netProfit, 4.61002317005007);
  assert.ok(lpPutSpreadSnapshot(100.27, 100, 5).netProfit < 0);
  for (const width of [5, 25, 100]) {
    const { longRange } = lpPutSpreadSnapshot(100, 100, width);
    const low = lpPutSpreadSnapshot(Math.min(80, longRange.lower) / 2, 100, width);
    const high = lpPutSpreadSnapshot(Math.max(125, longRange.upper) * 2, 100, width);
    close(low.netEth, 0);
    close(low.netUsdc, 0);
    close(high.netEth, 0);
    close(high.netUsdc, 0);
    close(low.netProfit, high.netProfit);
  }
});

test('Range comparison balances, slopes, and boundary continuity agree for all widths', () => {
  for (const width of [5, 10, 25, 50, 100]) {
    const { longRange } = lpPutSpreadSnapshot(100, 100, width);
    for (const price of [30, 70, 90, 99.3, 100.7, 111, 140, 175, 225]) {
      const step = 0.001;
      const before = lpPutSpreadSnapshot(price - step, 100, width);
      const at = lpPutSpreadSnapshot(price, 100, width);
      const after = lpPutSpreadSnapshot(price + step, 100, width);
      close(at.netValue, at.netEth * price + at.netUsdc);
      close(at.netProfit, at.shortProfit + at.longProfit);
      close((after.netProfit - before.netProfit) / (2 * step), at.netEth);
      close((after.netEth - before.netEth) / (2 * step), at.netGamma);
    }
    for (const bound of [80, 125, longRange.lower, longRange.upper]) {
      const before = lpPutSpreadSnapshot(bound - 0.000001, 100, width);
      const after = lpPutSpreadSnapshot(bound + 0.000001, 100, width);
      close(before.netProfit, after.netProfit, 0.00001);
      close(before.netEth, after.netEth, 0.00001);
      close(before.netUsdc, after.netUsdc, 0.001);
    }
  }
  for (const width of [0, -1, NaN, Infinity]) assert.throws(() => lpPutSpreadSnapshot(100, 100, width), RangeError);
});
