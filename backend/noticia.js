const mongoose = require('mongoose');

const NoticiaSchema = new mongoose.Schema({
   _id: { type: String, required: true },
    manchete: { type: String, required: true },
    lide: { type: String },
    portal: {nome: { type: String }},
    data: { type: Date, required: true },
    autor: [{ nome: { type: String }}],
    sSentimento: { type: Number }, // Assumindo -1, 0, 1
    assunto: [{ TAG: { type: String }
    }],
}, { collection: 'Noticias', _id: false });

const Noticia = mongoose.model('Noticia', NoticiaSchema);

module.exports = Noticia;