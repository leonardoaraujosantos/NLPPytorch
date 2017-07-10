#include "conceptmatcher.h"

ConceptMatcher::ConceptMatcher()
{

}

void ConceptMatcher::AddConcept(std::string concept)
{
    std::transform(concept.begin(), concept.end(), concept.begin(), ::tolower);
    m_set_concepts.insert(concept);
}

list<string> ConceptMatcher::GetTokens(std::string input)
{
    // Regular expression to filter text
    regex re("[^0-9a-zA-Z]+");
    list<string> retList;

    // Transform input to lowercase
    std::transform(input.begin(), input.end(), input.begin(), ::tolower);

    // Tokenize text, filtering and adding results to list
    std::copy( std::sregex_token_iterator(input.begin(), input.end(), re, -1),
                      std::sregex_token_iterator(),
                      std::back_inserter(retList));

    return retList;
}

// Look for matches between the list of tokens from the input and the concepts
list<string> ConceptMatcher::GetMatches(string input)
{
    list<string> retList;
    list<string> inTokens = GetTokens(input);

    // Iterate on input tokens looking for a concept
    for (const auto &token : inTokens) {
        // Searching on a hash table is O(1) on average
        auto search = m_set_concepts.find(token);
            if(search != m_set_concepts.end()) {
                // Concept found add on retList
                retList.push_back(token);
                //std::cout << "Found " << (*search) << '\n';
            }
    }
    return retList;
}
