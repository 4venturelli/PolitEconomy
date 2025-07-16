const {insereBD} = require("./BD_Insere")

async function insereNoticias(jsons, db) {
  try {
    await insereBD(jsons, db)
  } catch (err) {
    if (err.name === 'MongoBulkWriteError' || err.code === 11000) {
      const totalErros = err.writeErrors ? err.writeErrors.length : 0
            
      if ((totalErros / jsons.length) >= 0.5) {
        return 1 - totalErros / jsons.length
      } 
    } 
  }
  return 1
}

module.exports = {insereNoticias}