function processaFiltros(query) {
  const filtros = {
    pagina: 1,
    quantidade: 10
  }
  if(Object.keys(query).length > 0) {
    if(Object.hasOwnProperty.bind(query)('pagina')) {
      let pagina = parseInt(query.pagina)
      if(isNaN(pagina) || pagina < 1) {
        throw new Error("ERRO: valor de página inválido!")
      }
      filtros['pagina'] = pagina
    } 
    if(Object.hasOwnProperty.bind(query)('campo')) {
      let campo
      if(query.campo.toLowerCase() === 'tópico') {
        campo = 'assunto.TAG'
      } else if (query.campo.toLowerCase() === 'autor') {
        campo = 'autor.nome'
      } else if (query.campo.toLowerCase() === 'manchete') {
        campo = 'manchete'
      } else {
        throw new Error("ERRO: o campo de busca fornecido é inexistente ou inválido.")
      }
      filtros['campo'] = campo
    }
    if(Object.hasOwnProperty.bind(query)('dataInicio')) {
      let dataInicio = (!isNaN(new Date(query.dataInicio))) ? new Date(query.dataInicio) : null
      if(dataInicio !== null) {
        filtros['dataInicio'] = dataInicio
      }
    }

    if(Object.hasOwnProperty.bind(query)('dataLimite')) {
      let dataLimite = (!isNaN(new Date(query.dataLimite))) ? new Date(query.dataLimite) : null
      if(dataLimite !== null) {
        filtros['dataLimite'] = dataLimite
      } 
    }

    if(Object.hasOwnProperty.bind(query)('portal')) {
      let portal = query.portal
      if(portal != ''){
        filtros['portal'] = portal
      }
    }

    if(Object.hasOwnProperty.bind(query)('quantidade')) {
      let quantidade = query.quantidade
      quantidade = parseInt(quantidade)
      if(!isNaN(quantidade)) {
        filtros['quantidade'] = quantidade
      }
    }
  } else {
    throw new Error('ERRO: não foram especificados parametros de filtros na busca.')
  }
  return filtros
}

module.exports ={
  processaFiltros
}