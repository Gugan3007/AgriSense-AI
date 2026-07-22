import test from 'node:test';
import assert from 'node:assert/strict';

import { CROP_OPTIONS, isConclusive, uploadErrorDetails } from '../src/utils/analysis.js';

test('maps non-leaf response into actionable blocked state', () => {
  const result = uploadErrorDetails({ response: { data: {
    status: 'rejected', reason_code: 'non_leaf',
    message: 'This does not look like a leaf.', guidance: ['Upload one leaf.'],
  } } });

  assert.equal(result.status, 'rejected');
  assert.equal(result.reasonCode, 'non_leaf');
  assert.deepEqual(result.guidance, ['Upload one leaf.']);
});

test('does not treat inconclusive analysis as a prediction', () => {
  assert.equal(isConclusive({ analysis_status: 'inconclusive' }), false);
  assert.equal(isConclusive({ analysis_status: 'completed' }), true);
});

test('submits crop values that exactly match the model contract', () => {
  assert.deepEqual(CROP_OPTIONS.map(({ value }) => value), [
    'Apple', 'Cherry_(including_sour)', 'Corn_(maize)', 'Grape', 'Peach',
    'Pepper,_bell', 'Potato', 'Strawberry', 'Tomato',
  ]);
});
