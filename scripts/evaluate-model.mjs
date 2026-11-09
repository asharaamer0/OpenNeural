import fs from 'node:fs';
import {BENCHMARK_TRACKS,createRoad} from '../dist/tracks.js';
import {validateModel} from '../dist/network.js';
import {evaluateModel} from '../dist/evaluate.js';
const model=validateModel(JSON.parse(fs.readFileSync('dist/models/baseline.json','utf8')));const report=evaluateModel(model,BENCHMARK_TRACKS.map(createRoad));fs.writeFileSync('dist/models/evaluation.json',JSON.stringify(report));for(const t of report.tracks)console.log(JSON.stringify({track:t.track,passed:t.passed,total:t.total,nominal:{...t.nominal,trace:undefined},perturbed:t.perturbed.map(a=>({...a,trace:undefined}))}));
