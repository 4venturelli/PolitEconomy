const { conectaBD, desconectaBD } = require("./BD_Conexao")
const { insereNoticias } = require("./insereNoticias")

module.exports = {
  conectaBD,
  desconectaBD,
  insereNoticias,
}