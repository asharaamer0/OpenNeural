import fs from 'node:fs';
import {evaluateTraffic} from '../dist/road-evaluate.js';
const model=JSON.parse(fs.readFileSync('dist/models/road-policy.json','utf8'));
const report=evaluateTraffic(model,{onProgress:({row})=>console.log(JSON.stringify(row))});
fs.writeFileSync('dist/models/road-evaluation.json',JSON.stringify(report,null,2));
