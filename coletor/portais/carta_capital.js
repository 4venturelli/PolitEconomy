const puppeteer = require('puppeteer')
const { conectaBD, desconectaBD, insereNoticias } = require('./BD_handler')


async function coletaDadosCartaCapital(pagina, link) {
  await pagina.goto(link)
  const noticia = await pagina.evaluate(() => {
    let paywall = document.querySelector(".paywall.active")
    if(paywall) return null;
    let dados = {
      portal: {nome:"Carta Capital"},
      _id: window.location.href
    }

    // Manchete
    let manchete = document.querySelector("section.s-content__heading h1")
    if(manchete) {
      dados.manchete = manchete.textContent
    } else {
      return null
    }

    // Lide
    let lide = Array.from(document.querySelectorAll("section.s-content__heading p")).map(x => x.textContent)
    if(lide.length > 0) {
      lide = lide.filter(x => x.length > 0)
      dados.lide = lide[0].trim()
    } 

    // Data
    let dataPublicacao = document.querySelector("meta[property='article:published_time']")
    if(dataPublicacao) {
      dados.data = dataPublicacao.getAttribute('content')
    } else {
      return null
    }

    // Autores
    let autores = Array.from(document.querySelectorAll("div.s-content__infos strong")).map(x => x.textContent.trim())
    dados.autor = autores.map(x => {
      let autor = {nome: x}
      return autor
    })

    // Tags
    let tags = Array.from(document.querySelectorAll("meta[property='article:tag']")).map(x => x.getAttribute('content').toUpperCase())
    if(tags){
      dados.assunto = tags.map(x => {
        let tag = {TAG: x}
        return tag
      })
    }

    // // Artigo 
    // let artigo = Array.from(document.querySelectorAll(".s-content__text .content-closed p")).map(x => x.innerText.trim())
    // artigo = artigo.filter(x => x.length > 0)
    // if(artigo.length > 0) {
    //   dados.artigo = artigo.map(x => x.replaceAll(/\\n/g, '\n'))
    // }

    return dados
  })
  if(noticia != null) {
    noticia.data = new Date(noticia.data)
    if((noticia.data instanceof Date) == false) noticia = null
  }
  return noticia
}


async function scrapCartaCapital(URL, tipo) {
  const browser = await puppeteer.launch({headless: true})
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD()

  let dict = []

  try {

    for (let pagina = 1; pagina <= 100; pagina++) {
      let cartaCapitalURL = `${URL}${pagina}/`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(cartaCapitalURL, { waitUntil: "domcontentloaded" })

      const links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll("a.l-list__item")).map(x => x.getAttribute("href"))
      })

    
      await scrapingPage.bringToFront()
      for (let i = 0; i < links.length; i++) {
        let temp = await coletaDadosCartaCapital(scrapingPage, links[i])
        
        if(temp == null) continue;
        temp.secao = tipo
        dict.push(temp)
      }

        let inseridos = await insereNoticias(dict, db)
        if(inseridos < 0.5) {
          return null
        }
        dict = []
      }
    if(dict.length > 0) await insereNoticias(dict, db)
      
  } catch (err) {
    if(dict.length > 0) await insereNoticias(dict, db)
    return
    console.error("Erro:", err)
  } finally {

    await desconectaBD(client)
    await scrapingPage.close()
    await browser.close()
  }
}

async function scrapingCartaCapital(){
  console.log('Coletando Carta Capital...')
  await Promise.all([
    scrapCartaCapital("https://www.cartacapital.com.br/economia/page/", "Economia")
  ])



  // await scrapCartaCapital("https://www.cartacapital.com.br/politica/page/", "Política")
}

module.exports = {scrapingCartaCapital}