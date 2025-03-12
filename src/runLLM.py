
import sys
import os
import warnings

import streamlit as sl
from pathlib import Path
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

from src.getCorpus import getCorpusSTJ, getCorpusSTF

warnings.filterwarnings("ignore")
sys.path.append(str(Path(__file__).parent.parent.parent)) 

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def load_prompt():
    
    corpus = getCorpusSTF()
    
    # prompt = """Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, que busca assinalar os teses com ou sem repercussao geral (temas) do STF ou STJ mais relevantes de cada processo.
    #     Lista ordenadamente por relevencia as teses ou temas com ou sem repercussão geral mais relevantes em portugues.
    # """    
    
    prompt = """ Voce é um software especialista em assuntos juridicos, focado em analise de processos e recursos, 
        que busca assinalar os temas STF ou STJ mais relevantes de cada processo.
        Contexto = {context}
        Pergunta = {question}.
        Lista ordenadamente por relevencia os temas mais relevantes em portugues:
    """
    
    teste = """ Com base na lista de temas do STF/STJ abaixo, analise o seguinte documento e identifique a quais temas ele mais se assemelha. Considere a relação de conteúdo, 
    jurisprudência aplicável e palavras-chave presentes no texto. Liste os temas mais relevantes e explique brevemente o motivo da correspondência.
    
    Lista de temas: 
    
    Tema 578 - Liquidação / Cumprimento / Execução - Em princípio, nos termos do art. 9°, III, da Lei 6.830/1980, cumpre ao executado nomear bens à penhora, observada a ordem legal. É dele o ônus de comprovar a imperiosa necessidade de afastá-la, e, para que essa providência seja adotada, mostra-se insufici

Tema 514 - DIREITO CIVIL, Previdência privada, Espécies de Contratos, Resgate de Contribuição - A quitação relativa à restituição, por instrumento de transação, somente alcança as parcelas efetivamente quitadas, não tendo eficácia em relação às verbas por ele não abrangidas. Portanto, se os expurgos inflacionários não foram pagos aos participantes q

Tema 723 - Liquidação / Cumprimento / Execução - A sentença proferida pelo Juízo da 12ª Vara Cível da Circunscrição Especial Judiciária de Brasília/DF, na ação civil coletiva nº 1998.01.1.016798-9, que condenou o Banco do Brasil ao pagamento de diferenças decorrentes de expurgos inflacionários sobre cad

Tema 568 - Extinção da Execução - A efetiva constrição patrimonial e a efetiva citação (ainda que por edital) são aptas a interromper o curso da prescrição intercorrente, não bastando para tal o mero peticionamento em juízo, requerendo, v.g., a feitura da penhora sobre ativos financeiros

Tema 988 - Cabimento - O rol do art. 1.015 do CPC é de taxatividade mitigada, por isso admite a interposição de agravo de instrumento quando verificada a urgência decorrente da inutilidade do julgamento da questão no recurso de apelação.

Tema 880 - Prescrição e Decadência - "A partir da vigência da Lei n. 10.444/2002, que incluiu o § 1º ao art. 604, dispositivo que foi sucedido, conforme Lei n. 11.232/2005, pelo art. 475-B, §§ 1º e 2º, todos do CPC/1973, não é mais imprescindível, para acertamento da conta exequenda, a junta

Tema 481 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Competência , Órgãos Judiciários e Auxiliares da Justiça, Processo e Procedimento, Liquidação / Cumprimento / Execução - A sentença genérica proferida na ação civil coletiva ajuizada pela Apadeco, que condenou o Banestado ao pagamento dos chamados expurgos inflacionários sobre cadernetas de poupança, dispôs que seus efeitos alcançariam todos os poupadores da instituição fin

Tema 391 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Competência - O juízo do inventário, na modalidade de arrolamento sumário, não detém competência para apreciar pedido de reconhecimento da isenção do ITCMD (Imposto sobre Transmissão Causa Mortis e Doação de quaisquer Bens ou Direitos), à luz do disposto no caput do ar

Tema 1004 - Desapropriação Indireta - Reconhecida a incidência do princípio da boa-fé objetiva em ação de desapropriação indireta, se a aquisição do bem ou de direitos sobre ele ocorrer quando já existente restrição administrativa, fica subentendido que tal ônus foi considerado na fixação do

Tema 398 - DIREITO TRIBUTÁRIO, ISS/ Imposto sobre Serviços, Repetição de indébito - A pretensão repetitória de valores indevidamente recolhidos a título de ISS incidente sobre a locação de bens móveis (cilindros, máquinas e equipamentos utilizados para acondicionamento dos gases vendidos), hipótese em que o tributo assume natureza indire

Tema 671 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Liquidação / Cumprimento / Execução, Honorários Periciais - Na liquidação por cálculos do credor, descabe transferir do exequente para o executado o ônus do pagamento de honorários devidos ao perito que elabora a memória de cálculos.

Tema 922 - Indenização por Dano Moral - A inscrição indevida comandada pelo credor em cadastro de inadimplentes, quando preexistente legítima anotação, não enseja indenização por dano moral, ressalvado o direito ao cancelamento. Inteligência da Súmula 385/STJ.

Tema 697 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Prazo, Intimação - A ausência da cópia da certidão de intimação da decisão agravada não é óbice ao conhecimento do Agravo de Instrumento quando, por outros meios inequívocos, for possível aferir a tempestividade do recurso, em atendimento ao princípio da instrumentalidade d

Tema 890 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Liquidação / Cumprimento / Execução - Na execução individual de sentença proferida em ação civil pública que reconhece o direito de poupadores aos expurgos inflacionários decorrentes do Plano Verão (janeiro de 1989), descabe a inclusão de juros remuneratórios nos cálculos de liquidação se ine

Tema 706 - Multa Cominatória / Astreintes - A decisão que comina astreintes não preclui, não fazendo tampouco coisa julgada.

Tema 173 - DIREITO PROCESSUAL CIVIL E DO TRABALHO - O 'contribuinte de fato' (in casu, distribuidora de bebida) não detém legitimidade ativa ad causam para pleitear a restituição do indébito relativo ao IPI incidente sobre os descontos incondicionais, recolhido pelo 'contribuinte de direito' (fabricante de

Tema 133 - DIREITO PROCESSUAL CIVIL E DO TRABALHO - A autenticação de cópias do Agravo de Instrumento do artigo 522, do CPC, resulta como diligência não prevista em lei, em face do acesso imediato aos autos principais, propiciado na instância local. A referida providência somente se impõe diante da impugna

Tema 526 - Liquidação / Cumprimento / Execução - A atribuição de efeitos suspensivos aos embargos do devedor" fica condicionada "ao cumprimento de três requisitos: apresentação de garantia; verificação pelo juiz da relevância da fundamentação (fumus boni juris) e perigo de dano irreparável ou de difícil

Tema 92 - DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO, Energia Elétrica - As OBRIGAÇÕES AO PORTADOR emitidas pela ELETROBRÁS em razão do empréstimo compulsório instituído pela Lei 4.156/62 não se confundem com as DEBÊNTURES e, portanto, não se aplica a regra do art. 442 do CCom, segundo o qual prescrevem em 20 anos as ações fun

Tema 268 - DIREITO TRIBUTÁRIO - É desnecessária a apresentação do demonstrativo de cálculo, em execução fiscal, uma vez que a Lei n. 6.830/80 dispõe, expressamente, sobre os requisitos essenciais para a instrução da petição inicial e não elenca o demonstrativo de débito entre eles.

Tema 529 - DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO, Inquérito / Processo / Recurso Administrativo, Gratificação Incorporada / Quintos e Décimos / VPNI - No caso, o direito à incorporação dos quintos surgiu com a edição da MP n. 2.225-45/2001. Portanto, em 04 de setembro de 2001, quando publicada a MP, teve início o prazo prescricional quinquenal do art. 1º do Decreto 20.910/32. A prescrição foi interrompi

Tema 333 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Liquidação / Cumprimento / Execução - Na oportunidade da liquidação da sentença, por se tratar de reconhecimento de crédito-prêmio de IPI, a parte deverá apresentar toda a documentação suficientes à comprovação da efetiva operação de exportação, bem como do ingresso de divisas no País, sem o

Tema 1132 - Alienação Fiduciária - Em ação de busca e apreensão fundada em contratos garantidos com alienação fiduciária (art. 2º, § 2º, do Decreto-Lei n. 911/1969), para a comprovação da mora, é suficiente o envio de notificação extrajudicial ao devedor no endereço indicado no instrumento

Tema 733 - Controle de Preços - A eficácia da Lei 4.870/1965, que previa a sistemática de tabelamento de preços promovida pelo IAA, findou em 31/01/1991, em virtude da publicação, em 01/02/1991, da Medida Provisória 295, de 31/01/1991, posteriormente convertida na Lei 8.178, de 01/03/19

Tema 409 - Honorários Advocatícios - Em caso de sucesso da impugnação, com extinção do feito mediante sentença (art. 475-M, § 3º), revela-se que quem deu causa ao procedimento de cumprimento de sentença foi o exequente, devendo ele arcar com as verbas advocatícias.

Tema 374 - DIREITO TRIBUTÁRIO, IPI/ Imposto sobre Produtos Industrializados, Base de Cálculo - A dedução dos descontos incondicionais é vedada, no entanto, quando a incidência do tributo se dá sobre valor previamente fixado, nos moldes da Lei 7.798/89 (regime de preços fixos), salvo se o resultado dessa operação for idêntico ao que se chegaria com

Tema 627 - DIREITO PREVIDENCIÁRIO, Benefícios em Espécie, Auxílio-Acidente (Art. 86), Pedidos Genéricos Relativos aos Benefícios em Espécie, Concessão - O segurado especial, cujo acidente ou moléstia é anterior à vigência da Lei n. 12.873/2013, que alterou a redação do inciso I do artigo 39 da Lei n. 8.213/91, não precisa comprovar o recolhimento de contribuição como segurado facultativo para ter direito

Tema 1078 - Indenização por Dano Moral - O atraso, por parte de instituição financeira, na baixa de gravame de alienação fiduciária no registro de veículo não caracteriza, por si só, dano moral in re ipsa.

Tema 506 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Recurso, Liquidação / Cumprimento / Execução, Honorários Advocatícios - Hipótese de ocorrência da preclusão lógica a que se refere o legislador no art. 503 do CPC, segundo o qual 'A parte, que aceitar expressa ou tacitamente a sentença ou a decisão, não poderá recorrer'. Isso porque, apesar da expressa postulação de arbitrame

Tema 564 - Cheque - Em ação monitória fundada em cheque prescrito, ajuizada em face do emitente, é dispensável menção ao negócio jurídico subjacente à emissão da cártula.

Tema 560 - Enriquecimento sem Causa - Em se tratando de pedido relativo a valores para cujo ressarcimento não havia previsão contratual (pactuação prevista em instrumento, em regra, nominado de 'TERMO DE CONTRIBUIÇÃO'), a pretensão prescreve em 20 (vinte) anos, na vigência do Código Civil de

Tema 1218 - Princípio da Insignificância - A reiteração da conduta delitiva obsta a aplicação do princípio da insignificância ao crime de descaminho - independentemente do valor do tributo não recolhido -, ressalvada a possibilidade de, no caso concreto, se concluir que a medida é socialmente reco

Tema 887 - Expurgos Inflacionários / Planos Econômicos - Na execução individual de sentença proferida em ação civil pública que reconhece o direito de poupadores aos expurgos inflacionários decorrentes do Plano Verão (janeiro de 1989): (I) descabe a inclusão de juros remuneratórios nos cálculos de liquidação se

Tema 1036 - DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO, Liberação de Veículo Apreendido - "A apreensão do instrumento utilizado na infração ambiental, fundada na atual redação do § 4º do art. 25 da Lei 9.605/1998, independe do uso específico, exclusivo ou habitual para a empreitada infracional".

Tema 891 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Liquidação / Cumprimento / Execução, Valor da Execução / Cálculo / Atualização, Correção Monetária - Na execução de sentença que reconhece o direito de poupadores aos expurgos inflacionários decorrentes do Plano Verão (janeiro de 1989), incidem os expurgos inflacionários posteriores a título de correção monetária plena do débito judicial, que terá como b

Tema 1043 - DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO, Liberação de Veículo Apreendido - O proprietário do veículo apreendido em razão de infração de transporte irregular de madeira não titulariza direito público subjetivo de ser nomeado fiel depositário do bem, as providências dos arts. 105 e 106 do Decreto Federal n. 6.514/2008 competindo a

Tema 410 - Honorários Advocatícios - O acolhimento ainda que parcial da impugnação gerará o arbitramento dos honorários, que serão fixados nos termos do art. 20, § 4º, do CPC, do mesmo modo que o acolhimento parcial da exceção de pré-executividade, porquanto, nessa hipótese, há extinção tamb

Tema 482 - Processo e Procedimento - A sentença genérica prolatada no âmbito da ação civil coletiva, por si, não confere ao vencido o atributo de devedor de 'quantia certa ou já fixada em liquidação' (art. 475-J do CPC), porquanto, 'em caso de procedência do pedido, a condenação será genéric

Tema 472 - DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO, Desapropriação - O depósito judicial do valor simplesmente apurado pelo corpo técnico do ente público, sendo inferior ao valor arbitrado por perito judicial e ao valor cadastral do imóvel, não viabiliza a imissão provisória na posse.

Tema 531 - DIREITO ADMINISTRATIVO E OUTRAS MATÉRIAS DE DIREITO PÚBLICO, Servidor Público Civil, Gratificações da Lei 8.112/1990 - Quando a Administração Pública interpreta erroneamente uma lei, resultando em pagamento indevido ao servidor, cria-se uma falsa expectativa de que os valores recebidos são legais e definitivos, impedindo, assim, que ocorra desconto dos mesmos, ante a boa-

Tema 761 - DIREITO TRIBUTÁRIO, IPI/ Imposto sobre Produtos Industrializados - Inexigibilidade do ressarcimento de custos e demais encargos pelo fornecimento de selos de controle de IPI instituído pelo DL 1.437/1975, que, embora denominado ressarcimento prévio, é tributo da espécie Taxa de Poder de Polícia, de modo que há vício de f

Tema 571 - Extinção da Execução - A Fazenda Pública, em sua primeira oportunidade de falar nos autos (art. 245 do CPC/73, correspondente ao art. 278 do CPC/2015), ao alegar nulidade pela falta de qualquer intimação dentro do procedimento do art. 40 da LEF, deverá demonstrar o prejuízo que

Tema 676 - Custas - Não se determina o cancelamento da distribuição se o recolhimento das custas, embora intempestivo, estiver comprovado nos autos.

Tema 425 - DIREITO PROCESSUAL CIVIL E DO TRABALHO, Liquidação / Cumprimento / Execução - A utilização do Sistema BACEN-JUD, no período posterior à vacatio legis da Lei 11.382/2006 (21.01.2007), prescinde do exaurimento de diligências extrajudiciais, por parte do exequente, a fim de se autorizar o bloqueio eletrônico de depósitos ou aplicações

Tema 509 - Liquidação / Cumprimento / Execução - Com a atual redação do art. 475-N, inc. I, do CPC, atribuiu-se 'eficácia executiva' às sentenças 'que reconhecem a existência de obrigação de pagar quantia'.

Tema 721 - Honorários Advocatícios em Execução Contra a Fazenda Pública - A renúncia ao valor excedente ao previsto no art. 87 do ADCT, manifestada após a propositura da demanda executiva, não autoriza o arbitramento dos honorários, porquanto, à luz do princípio da causalidade , a Fazenda Pública não provocou a instauração da E

Tema 905 - Juros de Mora - Legais / Contratuais - 1. Correção monetária: o art. 1º-F da Lei 9.494/97 (com redação dada pela Lei 11.960/2009), para fins de correção monetária, não é aplicável nas condenações judiciais impostas à Fazenda Pública, independentemente de sua natureza.1.1 Impossibilidade de fix

Tema 520 - Mútuo - Tratando-se de contrato de mútuo para aquisição de imóvel garantido pelo FCVS, avençado até 25/10/96 e transferido sem a interveniência da instituição financeira, o cessionário possui legitimidade para discutir e demandar em juízo questões pertinentes às

Tema 230 - DIREITO PROCESSUAL CIVIL E DO TRABALHO - O recurso de apelação devolve, em profundidade, o conhecimento da matéria impugnada, ainda que não resolvida pela sentença, nos termos dos parágrafos 1º e 2º do art. 515 do CPC, aplicável a regra iura novit curia. Consequentemente, o Tribunal a quo pode s

Tema 570 - Extinção da Execução - A Fazenda Pública, em sua primeira oportunidade de falar nos autos (art. 245 do CPC/73, correspondente ao art. 278 do CPC/2015), ao alegar nulidade pela falta de qualquer intimação dentro do procedimento do art. 40 da LEF, deverá demonstrar o prejuízo que
    
    """
    
    prompt_template = """
        Regra:  Voce é um software especialista em assuntos juridicos, 
        focado em analise de processos e recursos, 
        que busca assinalar os temas STF ou STJ mais relevantes de cada processo.
        Caso você não tenha informações relevantes, retorne 'Desculpe não consegui achar uma resolução para a sua questão".
        Use os parametros abaixo para recuperar o contexto para a resposta.
        Questão: {query}
        Contexto: {context}
    """
    
    
    
    prompt = ChatPromptTemplate.from_template(prompt)
    
    return prompt

def load_llm():
    
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    return llm


if __name__ == '__main__':    
       
    # call main function
    load_prompt()