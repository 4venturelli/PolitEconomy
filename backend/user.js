const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    name: { type: String, required: true },
    email: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    preferences: {
        keywords: [{ type: String }], 
        
    },
    lastNotificationCheck: { 
        type: Date,
        default: Date.now // Define a data de criação como padrão
    }
});

const User = mongoose.model('User', UserSchema);

module.exports = mongoose.model('User', UserSchema, 'Usuarios');