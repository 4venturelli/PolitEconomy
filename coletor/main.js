const{scrapingAgenciaBrasil} = require('./portais/agencia_brasil')
const{scrapingCartaCapital} = require('./portais/carta_capital')
const{scrapingRevistaForum} = require('./portais/revista_forum')
const{scrapingG1} = require('./portais/g1')
const{scrapingCongressoEmFoco} = require('./portais/congresso')
const{scrapingCamaraDeputados} = require('./portais/camara_deputados')
const{scrapingInfoMoney} = require('./portais/infoMoney')
const{scrapingTempo} = require('./portais/tempo')
const{scrapingBBC} = require('./portais/bbc')

// const{conectar, desconectar} = require('./banco_de_dados/bancoConnection')

async function main(){
    await scrapingBBC()
    await scrapingCamaraDeputados()
    await scrapingG1()
    await scrapingTempo()
    await scrapingAgenciaBrasil()
    await scrapingCartaCapital()
    await scrapingCongressoEmFoco()
    await scrapingInfoMoney()
    await scrapingRevistaForum() 
}

main()