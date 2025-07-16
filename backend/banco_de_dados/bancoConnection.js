const { MongoClient } = require('mongodb')
require('dotenv').config()

const uri = process.env.MONGO_URI

async function conectar() {
    try {
        const client = new MongoClient(uri)
        await client.connect()
        const db = client.db("Banco_Coletor")
        return { db, client } // Retorna ambos
    } catch (err) {
        throw err
    }
}


async function desconectar(client) {
    try {
        await client.close()
    } catch (err) {
        throw err
    }
}



module.exports = { conectar, desconectar }
