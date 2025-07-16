const puppeteer = require('puppeteer')
const { conectaBD, desconectaBD, insereNoticias } = require("./BD_handler")

async function coletaDadosCongressoEmFoco(pagina, link) {
  await pagina.goto(link, {waitUntil: "domcontentloaded"})
  const noticia = await pagina.evaluate(() => {
    let dados = {
      portal: {nome:"Congresso em Foco"},
      _id: window.location.href
    }

    // Manchete
    let manchete = document.querySelector("h1.asset__title")
    if(manchete) {
      dados.manchete = manchete.textContent
    } else {
      return null
    }

    // Lide
    let lide = document.querySelector("h2.asset__summary")
    if(lide) dados.lide = lide.textContent

    // Data
    let dataPublicacao = document.querySelector("meta[name='published']")
    if(dataPublicacao) {
      dados.data = dataPublicacao.getAttribute('content')
    } else {
      return null
    }
    // Autores
    let autores = Array.from(document.querySelectorAll("p.asset__publisher")).map(x => x.textContent)
    if(autores && autores.length > 0){
      dados.autor = autores.map(x => {
        let autor = {nome: x.trim()}
        return autor
      })
    }


    //Tags
    let tags = document.querySelector("meta[property='article:tag']")
    if(tags){
      let tagsTexto = tags.getAttribute('content')
      if((tagsTexto == "null") == false) {
        tagsTexto = tagsTexto.split(',')
        dados.assunto = tagsTexto.map(x => {
          let tag = {TAG: x.trim().toUpperCase()}
          return tag
        })
      }
    }
    return dados  
  })
  if(noticia != null) {
    noticia.data = new Date(noticia.data)
    if((noticia.data instanceof Date) == false) noticia = null
  }
  return noticia
}

async function scrapCongressoEmFoco(URL, secao) {
  const browser = await puppeteer.launch()
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD()

  let dict = []
  
  try {
    for (let pagina = init + 1550; pagina <= 2000; pagina++) {
      let congressoURL = `${URL}${pagina}`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(congressoURL, { waitUntil: "domcontentloaded" })

      const links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll("p.asset__title a")).map(x => x.getAttribute("href"))
      })
        
      await scrapingPage.bringToFront()
      
      for (let i = 0; i < links.length; i++) {
        let temp = await coletaDadosCongressoEmFoco(scrapingPage, links[i])
        if(temp == null) continue;
        temp.secao = secao
        dict.push(temp)
      }
      
      let inseridos = await insereNoticias(dict, db)
      if(inseridos < 0.5) {
        return null
      }
      dict = []
    }
  } catch (err) {
    return
    console.error("Erro:", err)
  } finally {
    await desconectaBD(client)
    await scrapingPage.close()
    await browser.close()
  }
}

async function scrapingCongressoEmFoco(){
  console.log('Coletando Congresso em Foco...')
  await scrapCongressoEmFoco("https://www.congressoemfoco.com.br/noticia?pagina=", "Política")
}

module.exports = {scrapingCongressoEmFoco}