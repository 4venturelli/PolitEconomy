async function insereBD(jsons, db) {
    try {
        const colecao = db.collection("Noticias")
        await colecao.insertMany(jsons, { ordered: false })
    } catch (err) {
        throw err
    }
}

module.exports = {insereBD}