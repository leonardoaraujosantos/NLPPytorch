#ifndef CONCEPTMATCHER_H
#define CONCEPTMATCHER_H

#include <string>
#include <regex>
#include <list>
#include <iostream>
#include <algorithm>
#include <unordered_set>

using namespace std;

class ConceptMatcher
{
public:
    ConceptMatcher();
    void AddConcept(std::string concept);
    list<string> GetTokens(std::string input);
    list<string> GetMatches(std::string input);
private:
    list<string> m_concepts;
    // hash table of keys
    std::unordered_set<std::string> m_set_concepts;
};

#endif // CONCEPTMATCHER_H
