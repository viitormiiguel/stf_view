
import tiktoken

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import textwrap
from time import monotonic

use_long_text = True

news_article = '''Trata-se de embargos de declaração opostos por BANCO DO BRASIL S/A, nos autos do cumprimento de sentença que lhe move JOAQUIM MANOEL
        GRAVATO GERALDES, em face do acórdão que julgou o agravo de instrumento nº  5287315-84.2023.8.21.7000/RS, assim ementado:
        AGRAVO DE INSTRUMENTO. NEGÓCIOS JURÍDICOS BANCÁRIOS. EXPURGOS INFLACIONÁRIOS. CÉDULA RURAL PIGNORATÍCIA. AÇÃO COLETIVA. CUMPRIMENTO DE SENTENÇA.
        VALOR DO LAUDO: R$ 1.240.303,39. FUNDAMENTO: CÉDULAS RURAIS NºS 89/00839-1 E 89/00875-8. ATUALIZAÇÃO MONETÁRIA. O TÍTULO JUDICIAL REFERIU QUE DEVEM SER 
        CORRIGIDOS MONETARIAMENTE OS VALORES A CONTAR DO PAGAMENTO A MAIOR PELOS ÍNDICES APLICÁVEIS AOS DÉBITOS JUDICIAIS (...), 
        QUESTÃO QUE NÃO PODE SER ALTERADA NESTA FASE, POIS ACOBERTADA PELA COISA JULGADA, NO CASO, O ENTENDIMENTO DESTE JULGADOR É DE QUE O ÍNDICE APLICÁVEL 
        AOS DÉBITOS JUDICIAIS É O IGP-M-FORO, POIS INDEXADOR QUE MELHOR REFLETE A CORROSÃO DA MOEDA PELO FENÔMENO INFLACIONÁRIO. A UTILIZAÇÃO DO 
        PROVIMENTO Nº 14/2022-CGJ, PUBLICADO EM 07/04/2022, SOMENTE SERÁ POSSÍVEL EM CARÁTER SUBSIDIÁRIO, OU SEJA, QUANDO INEXISTIR DEFINIÇÃO A 
        RESPEITO NOS AUTOS OU NA LEGISLAÇÃO, O QUE NÃO É O CASO DO PRESENTE FEITO.  NO PONTO, RECURSO DESPROVIDO. JUROS DE MORA – TERMO INICIAL. 
        MESMO EM EXECUÇÕES OU CUMPRIMENTOS DE SENTENÇA INDIVIDUAIS, OS JUROS DE MORA INCIDEM A PARTIR DA CITAÇÃO DO DEVEDOR NO PROCESSO DE CONHECIMENTO DA 
        AÇÃO CIVIL PÚBLICA QUANDO ESTA SE FUNDAR EM RESPONSABILIDADE CONTRATUAL, CUJO INADIMPLEMENTO JÁ PRODUZA A MORA, SALVO A CONFIGURAÇÃO DESTA EM MOMENTO ANTERIOR. 
        ENTENDIMENTO PACIFICADO EM SEDE DE JULGAMENTO REPETITIVO PELO SUPERIOR TRIBUNAL DE JUSTIÇA, NO RESP 1.370.899/SP (TEMA 685 DOS RECURSOS REPETITIVOS), 
        CUJA APLICAÇÃO DEVE SER OBSERVADA EM TODOS OS RECURSOS QUE VENTILEM A MESMA CONTROVÉRSIA. NO PONTO, RECURSO DESPROVIDO. AGRAVO DE INSTRUMENTO DESPROVIDO, POR UNANIMIDADE.
        (TJRS, AGRAVO DE INSTRUMENTO Nº 5287315-84.2023.8.21.7000, 24ª CÂMARA CÍVEL , DESEMBARGADOR JORGE MARASCHIN DOS SANTOS, POR UNANIMIDADE, JULGADO EM 29/11/2023)
        A parte embargante alega que há vícios na decisão recorrida. Sustenta que devem ser aplicados os indíces de correção dos débitos judiciais da Justiça Federal. 
        Argumenta que a aplicação do IGP-M não está prevista na decisão exequenda e acaba por violar a coisa julgada. Afirma que, em se tratando de 
        devedores solidários, não pode haver consequências diferentes sobre a mesma dívida. Pondera ser omisso o acórdão quanto ao fato de que a aplicação 
        do IGP-M implica em onerosidade excessiva ao devedor, bem como sobre a utilização do IPCA em todo o período. Manifesta que os juros de mora devem ser 
        contados desde a citação inicial em cada uma das liquidações e execuções individuais. Prequestiona os dispositivos legais invocados. Pede provimento.'''

model_name = "gpt-4o-mini"

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    model_name=model_name
)

texts = text_splitter.split_text(news_article)

docs = [Document(page_content=t) for t in texts]
print(len(docs))

OPENAI_API_KEY="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"

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)


prompt_template = """Write a concise summary of the following:

{text}

CONSCISE SUMMARY IN PORTUGUESE:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

num_tokens = num_tokens_from_string(news_article, model_name)
print(num_tokens)

gpt_35_turbo_max_tokens = 4097
verbose = True

if num_tokens < gpt_35_turbo_max_tokens:
  chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt, verbose=verbose)
else:
  chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=prompt, combine_prompt=prompt, verbose=verbose)

start_time = monotonic()
summary = chain.run(docs)

print(f"Chain type: {chain.__class__.__name__}")
print(f"Run time: {monotonic() - start_time}")
print(f"Summary: {textwrap.fill(summary, width=100)}")