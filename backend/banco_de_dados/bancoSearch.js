const { conectar, desconectar } = require('./bancoConnection')

async function assuntosEmAlta(quantidadeDias, quantidadeTopicos, secao) {
  const {db, client} = await conectar()
  let dia = new Date(Date.now() - quantidadeDias * 24 * 60 * 60 * 1000)
  const matchStage = {
    data: {$gte: dia}
  }
  if(secao != '') {
    matchStage['secao'] = secao
  }

  let query = await db.collection('Noticias').aggregate([
    {$match: matchStage},
    {$unwind: {
      path: '$assunto',
      includeArrayIndex: 'string',
      preserveNullAndEmptyArrays: false
    }},
    {$group: {
      _id: '$assunto.TAG',
      quantidadeNoticias: { $count: {} },
      dataUltimaAtualizacao: {$max: '$data'},
      mediaDeSentimento: {$avg: "$sSentimento"}
    }},
    {$sort: {quantidadeNoticias: -1}},
    {$limit: quantidadeTopicos}
  ])
  .toArray().then(dados => dados)

  await desconectar(client)
  return query;
}

async function sentimentosGerais(quantidadeDias, secao) {
  const {db, client} = await conectar()
  let dia = new Date(Date.now() - quantidadeDias * 24 * 60 * 60 * 1000)

  const matchStage = {
    data: {$gte: dia}
  }

  if(secao != '') {
    matchStage['secao'] = secao
  }

  const groupStage = {
    _id: {
      dia: {$dayOfMonth: '$data'},
      mes: {$month: '$data'},
      ano: {$year: '$data'}
    }
  }

  const sentimentos = await db.collection('Noticias').aggregate([
    {$match: matchStage },
    {
      $group: {

      ...groupStage,
      positivas: {
        $sum: {$cond: [{ $eq: ['$sSentimento', 1] }, 1, 0]}
      },
      neutras: {
        $sum: {$cond: [{ $eq: ['$sSentimento', 0] }, 1, 0]}
      },
      negativas: {
        $sum: {$cond: [{ $eq: ['$sSentimento', -1] }, 1, 0]}
      },
      total: {
        $count: {}
      },
    }
    },
    {$sort: {
      '_id.ano': 1,
      '_id.mes': 1,
      '_id.dia': 1
    }}
  ])
  .toArray()
  .then(dados => dados)
  
  await desconectar(client)
  return sentimentos
}

async function ultimasNoticias(dias, secao = ''){
    const {db, client} = await conectar()
    const colecao = db.collection('Noticias');
      
    const hoje = new Date()
    const dataLimite = new Date(hoje)
    dataLimite.setDate(hoje.getDate() - dias)
    let findStage = {
      data: {$gte: dataLimite}
    }
    if(secao != '') {
      findStage['secao'] = secao
    }

    const noticias = await colecao
        .find(findStage)
        .sort({ data: -1 })
        .limit(10) 
        .toArray()

    await desconectar(client)
    return noticias;

}

async function noticiasBuscaAvancada(tag, filtros) {
  const {db, client} = await conectar()
  const matchStage = {
    [filtros.campo]: { $regex: tag}
  };
  if (filtros.hasOwnProperty('dataInicio') || filtros.hasOwnProperty('dataLimite')) {
    matchStage.data = {};
    if (filtros.hasOwnProperty('dataInicio')) matchStage.data.$gte = new Date(filtros.dataInicio)
    if (filtros.hasOwnProperty('dataLimite')) matchStage.data.$lte = new Date(filtros.dataLimite)
  }

  if(filtros.hasOwnProperty('portal') && filtros.portal !== 'Todos') {
    matchStage['portal.nome'] = filtros.portal
  }

  let skip = (filtros.pagina - 1) * filtros.quantidade

  let query = await db.collection('Noticias').aggregate([
    {$match: matchStage},
    {$sort: {data: -1}},
    {$skip: skip},
    {$limit: filtros.quantidade}
  ])
  .toArray()
  .then(dados => dados)
  await desconectar(client)
  return query
}

async function sentimentosPortais(quantidadeDias, secao) {
  const {db, client} = await conectar()
  let dia = new Date(Date.now() - quantidadeDias * 24 * 60 * 60 * 1000)

  const matchStage = {
    data: {$gte: dia}
  }

  if(secao != '') {
    matchStage['secao'] = secao
  }

  const portais = await db.collection('Noticias').aggregate([
    {$match: matchStage },
    {
      $group: {
      _id: '$portal.nome',
      total: {
        $count: {}
      },
    }
    },
    {$sort: {
      '_id.ano': 1,
      '_id.mes': 1,
      '_id.dia': 1
    }}
  ])
  .toArray()
  .then(dados => dados)

  await desconectar(client)
  return portais
}

async function portaisDisponiveis() {
  const {db, client} = await conectar()
  let query = await db.collection('Noticias').aggregate([
    {$group: {_id: '$portal.nome'}},
    {$sort: {_id: 1}}
  ])
  .toArray()
  .then(dados => dados)
  await desconectar(client)
  return (query) ? query : null;
}

async function sentimentoAtravesDoTempo(tag, filtros) {
  const {db, client} = await conectar()
  const matchStage = {
    [filtros.campo]: { $regex: tag}
  };
  if (filtros.hasOwnProperty('dataInicio') || filtros.hasOwnProperty('dataLimite')) {
    matchStage.data = {};
    if (filtros.hasOwnProperty('dataInicio')) matchStage.data.$gte = new Date(filtros.dataInicio);
    if (filtros.hasOwnProperty('dataLimite')) matchStage.data.$lte = new Date(filtros.dataLimite);
  }

  if(filtros.hasOwnProperty('portal') && filtros.portal !== 'Todos') {
    matchStage['portal.nome'] = filtros.portal
  }

  const group = {
    _id: {
      mes: {$month: '$data'},
      ano: {$year: '$data'}
    }
  }
  if(filtros.hasOwnProperty('dataInicio') && filtros.hasOwnProperty('dataLimite')) {
    let diferencaDias = Math.abs(new Date(filtros.dataLimite) - new Date(filtros.dataInicio)) / (1000 * 3600 * 24)
    if(diferencaDias < 30) {
      group._id['dia'] = {$dayOfMonth: '$data'}
    }
  }

  let query = await db.collection('Noticias').aggregate(
    [
      { $match: matchStage},
      {
        $group: {
          ...group,
          positivas: {
            $sum: {$cond: [{ $eq: ['$sSentimento', 1] }, 1, 0]}
          },
          neutras: {
            $sum: {$cond: [{ $eq: ['$sSentimento', 0] }, 1, 0]}
          },
          negativas: {
            $sum: {$cond: [{ $eq: ['$sSentimento', -1] }, 1, 0]}
          }
        }
      },
      {$sort: {
        '_id.ano': 1,
        '_id.mes': 1,
        '_id.dia': 1
      }}
    ],
  )
  .toArray()
  .then(dados => dados)
  await desconectar(client)
  return query
}

module.exports = {
  assuntosEmAlta, 
  ultimasNoticias, 
  noticiasBuscaAvancada, 
  portaisDisponiveis,
  sentimentoAtravesDoTempo,
  sentimentosGerais,
  sentimentosPortais
}