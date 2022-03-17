import express from 'express';
import sprightly from 'sprightly';
import http from 'http' ;
import io from 'socket.io';
import fetch from 'node-fetch';
import { v4 as uuidv4 } from 'uuid';
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const upload = require('express-fileupload');
const path = require('path');

const app = express();
app.engine('spy', sprightly); 
app.set('views', './views'); 
app.set('view engine', 'spy'); 
app.use(express.static('./public')); // serve the "public" directory
app.use(upload());

const title = "NEVA";
const http_ = http.createServer(app);
const io_ = io(http_);
const PORT = process.env.PORT || 3000;

const backend = 'http:localhost:8050'

app.get('/query', async (req, res) => {
    const resource = req.query.resource;
    const response = await fetch(backend + '/' + resource, {
        method: 'GET',
        mode: 'cors',
        headers: { 'Content-Type': 'application/json' }
    });
    // TODO error handling
    const data = await response.json();
    
    return res.json(data);
})

app.get('/', (req, res) => {
    res.render('main/welcome.spy', { title, uuid: uuidv4() })
});

http_.listen(PORT, function () {
    console.log(`Server is listening on port http://localhost:${PORT}`);
});
