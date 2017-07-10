/*
 * Super vannila example of a string matcher, even
 * if I use some cool data structure (Hash table, trie, or suffix tree) will not scale as well
 * as a Sequence to Sequence Model.
 *
 * Some references:
 * http://bigocheatsheet.com/
 * http://www.cplusplus.com/reference/unordered_set/unordered_set/
 * https://github.com/philsquared/Catch/blob/master/docs/tutorial.md
 * */
#define CATCH_CONFIG_RUNNER
#define TEST true
#include "catch.hpp"

using namespace std;

// Run Tests
int runCatchTests()
{
    return Catch::Session().run();
}

int main()
{
    if (TEST)
    {
            return runCatchTests();
    }

    return 0;
}
