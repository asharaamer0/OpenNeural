import fs from 'node:fs';
import {TRACKS,createRoad} from '../dist/tracks.js';
import {createDemonstrations,splitDataset} from '../dist/dataset.js';
const data=createDemonstrations(createRoad(TRACKS[0]),{count:18000});const split=splitDataset(data);fs.mkdirSync('training',{recursive:true});fs.writeFileSync('training/dataset.json',JSON.stringify({...split,metadata:{seed:data.seed,tracks:data.tracks,cleanSamples:data.cleanSamples,perturbationSamples:data.perturbationSamples,reference:data.reference,augmentation:data.augmentation}}));console.log(JSON.stringify({training:split.trainX.length,validation:split.valX.length,reference:data.reference}));
