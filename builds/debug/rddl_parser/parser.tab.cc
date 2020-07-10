/* A Bison parser, made by GNU Bison 3.5.1.  */

/* Bison implementation for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2020 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Undocumented macros, especially those whose name start with YY_,
   are private implementation details.  Do not rely on them.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.5.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* First part of user prologue.  */
#line 1 "parser.ypp"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "logical_expressions.h"
#include "rddl.h"
#include "utils/system_utils.h"
#include "utils/timer.h"

#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_NO_UNPREFIXED_OPTIONS
#include "../doctest/doctest.h"

extern int yylex();
extern int yyparse();
typedef struct yy_buffer_state* YY_BUFFER_STATE;
extern YY_BUFFER_STATE yy_scan_string(const char * str);
extern void yy_delete_buffer(YY_BUFFER_STATE buffer);
extern void yyerror (std::string message);

std::string targetDir;
RDDLTask* rddlTask;


#line 101 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"

# ifndef YY_CAST
#  ifdef __cplusplus
#   define YY_CAST(Type, Val) static_cast<Type> (Val)
#   define YY_REINTERPRET_CAST(Type, Val) reinterpret_cast<Type> (Val)
#  else
#   define YY_CAST(Type, Val) ((Type) (Val))
#   define YY_REINTERPRET_CAST(Type, Val) ((Type) (Val))
#  endif
# endif
# ifndef YY_NULLPTR
#  if defined __cplusplus
#   if 201103L <= __cplusplus
#    define YY_NULLPTR nullptr
#   else
#    define YY_NULLPTR 0
#   endif
#  else
#   define YY_NULLPTR ((void*)0)
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Use api.header.include to #include this header
   instead of duplicating it here.  */
#ifndef YY_YY_HOME_ZIQI_PROST_MASTER_BUILDS_DEBUG_RDDL_PARSER_PARSER_TAB_HH_INCLUDED
# define YY_YY_HOME_ZIQI_PROST_MASTER_BUILDS_DEBUG_RDDL_PARSER_PARSER_TAB_HH_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    lessOrEqual_token = 258,
    greaterOrEqual_token = 259,
    negative_infinity_token = 260,
    equivalent_token = 261,
    imply_token = 262,
    equal_token = 263,
    nonEqual_token = 264,
    positive_infinity_token = 265,
    forall_token = 266,
    exists_token = 267,
    case_token = 268,
    if_token = 269,
    switch_token = 270,
    then_token = 271,
    else_token = 272,
    otherwise_token = 273,
    sum_token = 274,
    product_token = 275,
    kronDelta_token = 276,
    diracDelta_token = 277,
    uniform_token = 278,
    bernoulli_token = 279,
    discrete_token = 280,
    normal_token = 281,
    poisson_token = 282,
    exponential_token = 283,
    weibull_token = 284,
    gama_token = 285,
    dirichlet_token = 286,
    multinomial_token = 287,
    types_token = 288,
    variables_token = 289,
    cpfs_token = 290,
    cdfs_token = 291,
    reward_token = 292,
    domain_token = 293,
    requirements_token = 294,
    objects_token = 295,
    init_state_token = 296,
    state_action_constraints_token = 297,
    action_preconditions_token = 298,
    state_invariants_token = 299,
    instance_token = 300,
    non_fluents_token = 301,
    discount_token = 302,
    terminate_when_token = 303,
    horizon_token = 304,
    max_nondef_actions_token = 305,
    doubleNum_token = 306,
    id_token = 307,
    variable_token = 308,
    enum_token = 309,
    object_token = 310,
    integer_token = 311,
    real_token = 312,
    bool_token = 313,
    true_token = 314,
    false_token = 315,
    default_token = 316,
    level_token = 317,
    observ_fluent_token = 318,
    action_fluent_token = 319,
    state_fluent_token = 320,
    intermediate_token = 321,
    derived_fluent_token = 322,
    non_fluent_token = 323,
    intNum_token = 324,
    AgregateOperator = 325,
    NEGATIVE = 326
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 32 "parser.ypp"

    double d;
    int i;
    std::string* str;
    std::vector<std::string>* strs;

    RDDLTask* rddlTask;
    Parameter* parameter;
    std::vector<Parameter*>* parameters;
    ParameterList* parameterList;
    ParametrizedVariable* parametrizedVariable;
    std::vector<ParametrizedVariable*>* parametrizedVariables;
    LogicalExpression* logicalExpression;
    Type* type;
    std::vector<Type*>* types;
    std::vector<LogicalExpression*>* logicalExpressions;
    ConditionEffectPair* conditionEffect;
    std::vector<ConditionEffectPair*>* conditionEffects;
    DiscreteDistribution* lConstCaseList;

#line 246 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_HOME_ZIQI_PROST_MASTER_BUILDS_DEBUG_RDDL_PARSER_PARSER_TAB_HH_INCLUDED  */



#ifdef short
# undef short
#endif

/* On compilers that do not define __PTRDIFF_MAX__ etc., make sure
   <limits.h> and (if available) <stdint.h> are included
   so that the code can choose integer types of a good width.  */

#ifndef __PTRDIFF_MAX__
# include <limits.h> /* INFRINGES ON USER NAME SPACE */
# if defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stdint.h> /* INFRINGES ON USER NAME SPACE */
#  define YY_STDINT_H
# endif
#endif

/* Narrow types that promote to a signed type and that can represent a
   signed or unsigned integer of at least N bits.  In tables they can
   save space and decrease cache pressure.  Promoting to a signed type
   helps avoid bugs in integer arithmetic.  */

#ifdef __INT_LEAST8_MAX__
typedef __INT_LEAST8_TYPE__ yytype_int8;
#elif defined YY_STDINT_H
typedef int_least8_t yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef __INT_LEAST16_MAX__
typedef __INT_LEAST16_TYPE__ yytype_int16;
#elif defined YY_STDINT_H
typedef int_least16_t yytype_int16;
#else
typedef short yytype_int16;
#endif

#if defined __UINT_LEAST8_MAX__ && __UINT_LEAST8_MAX__ <= __INT_MAX__
typedef __UINT_LEAST8_TYPE__ yytype_uint8;
#elif (!defined __UINT_LEAST8_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST8_MAX <= INT_MAX)
typedef uint_least8_t yytype_uint8;
#elif !defined __UINT_LEAST8_MAX__ && UCHAR_MAX <= INT_MAX
typedef unsigned char yytype_uint8;
#else
typedef short yytype_uint8;
#endif

#if defined __UINT_LEAST16_MAX__ && __UINT_LEAST16_MAX__ <= __INT_MAX__
typedef __UINT_LEAST16_TYPE__ yytype_uint16;
#elif (!defined __UINT_LEAST16_MAX__ && defined YY_STDINT_H \
       && UINT_LEAST16_MAX <= INT_MAX)
typedef uint_least16_t yytype_uint16;
#elif !defined __UINT_LEAST16_MAX__ && USHRT_MAX <= INT_MAX
typedef unsigned short yytype_uint16;
#else
typedef int yytype_uint16;
#endif

#ifndef YYPTRDIFF_T
# if defined __PTRDIFF_TYPE__ && defined __PTRDIFF_MAX__
#  define YYPTRDIFF_T __PTRDIFF_TYPE__
#  define YYPTRDIFF_MAXIMUM __PTRDIFF_MAX__
# elif defined PTRDIFF_MAX
#  ifndef ptrdiff_t
#   include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  endif
#  define YYPTRDIFF_T ptrdiff_t
#  define YYPTRDIFF_MAXIMUM PTRDIFF_MAX
# else
#  define YYPTRDIFF_T long
#  define YYPTRDIFF_MAXIMUM LONG_MAX
# endif
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif defined __STDC_VERSION__ && 199901 <= __STDC_VERSION__
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned
# endif
#endif

#define YYSIZE_MAXIMUM                                  \
  YY_CAST (YYPTRDIFF_T,                                 \
           (YYPTRDIFF_MAXIMUM < YY_CAST (YYSIZE_T, -1)  \
            ? YYPTRDIFF_MAXIMUM                         \
            : YY_CAST (YYSIZE_T, -1)))

#define YYSIZEOF(X) YY_CAST (YYPTRDIFF_T, sizeof (X))

/* Stored state numbers (used for stacks). */
typedef yytype_int16 yy_state_t;

/* State numbers in computations.  */
typedef int yy_state_fast_t;

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef YY_ATTRIBUTE_PURE
# if defined __GNUC__ && 2 < __GNUC__ + (96 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_PURE __attribute__ ((__pure__))
# else
#  define YY_ATTRIBUTE_PURE
# endif
#endif

#ifndef YY_ATTRIBUTE_UNUSED
# if defined __GNUC__ && 2 < __GNUC__ + (7 <= __GNUC_MINOR__)
#  define YY_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
# else
#  define YY_ATTRIBUTE_UNUSED
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && ! defined __ICC && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN                            \
    _Pragma ("GCC diagnostic push")                                     \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")              \
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END      \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif

#if defined __cplusplus && defined __GNUC__ && ! defined __ICC && 6 <= __GNUC__
# define YY_IGNORE_USELESS_CAST_BEGIN                          \
    _Pragma ("GCC diagnostic push")                            \
    _Pragma ("GCC diagnostic ignored \"-Wuseless-cast\"")
# define YY_IGNORE_USELESS_CAST_END            \
    _Pragma ("GCC diagnostic pop")
#endif
#ifndef YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_BEGIN
# define YY_IGNORE_USELESS_CAST_END
#endif


#define YY_ASSERT(E) ((void) (0 && (E)))

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yy_state_t yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (YYSIZEOF (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (YYSIZEOF (yy_state_t) + YYSIZEOF (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYPTRDIFF_T yynewbytes;                                         \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * YYSIZEOF (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / YYSIZEOF (*yyptr);                        \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, YY_CAST (YYSIZE_T, (Count)) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYPTRDIFF_T yyi;                      \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  12
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   1582

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  96
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  63
/* YYNRULES -- Number of rules.  */
#define YYNRULES  237
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  767

#define YYUNDEFTOK  2
#define YYMAXUTOK   326


/* YYTRANSLATE(TOKEN-NUM) -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, with out-of-bounds checking.  */
#define YYTRANSLATE(YYX)                                                \
  (0 <= (YYX) && (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex.  */
static const yytype_int8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,    94,     2,    73,     2,
      92,    93,    79,    77,    86,    78,    88,    80,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,    87,    85,
      75,    84,    76,    91,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    89,     2,    90,    72,    95,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,    82,    71,    83,    74,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37,    38,    39,    40,    41,    42,    43,    44,
      45,    46,    47,    48,    49,    50,    51,    52,    53,    54,
      55,    56,    57,    58,    59,    60,    61,    62,    63,    64,
      65,    66,    67,    68,    69,    70,    81
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_int16 yyrline[] =
{
       0,   101,   101,   107,   108,   109,   110,   111,   112,   120,
     126,   127,   128,   129,   130,   131,   132,   133,   134,   135,
     136,   137,   138,   139,   140,   141,   142,   143,   148,   149,
     150,   151,   154,   155,   161,   164,   165,   169,   170,   171,
     175,   176,   177,   180,   181,   184,   185,   186,   187,   190,
     191,   194,   195,   196,   197,   198,   199,   205,   208,   209,
     212,   213,   214,   215,   216,   217,   218,   219,   222,   223,
     226,   227,   230,   231,   232,   233,   234,   235,   236,   237,
     238,   242,   243,   246,   247,   250,   251,   254,   255,   256,
     257,   260,   261,   268,   271,   272,   275,   276,   279,   282,
     296,   309,   310,   311,   314,   315,   318,   319,   320,   321,
     324,   325,   328,   329,   330,   331,   332,   340,   341,   342,
     343,   344,   346,   347,   348,   349,   350,   351,   352,   373,
     374,   375,   376,   377,   378,   379,   380,   381,   382,   383,
     384,   385,   386,   387,   388,   389,   390,   391,   392,   393,
     394,   395,   396,   397,   398,   401,   402,   403,   404,   405,
     406,   407,   408,   409,   410,   411,   414,   415,   418,   419,
     422,   423,   426,   435,   436,   439,   440,   444,   445,   446,
     452,   459,   460,   463,   464,   467,   473,   474,   477,   478,
     481,   487,   488,   491,   492,   495,   501,   505,   506,   509,
     516,   517,   518,   519,   527,   538,   548,   560,   561,   565,
     577,   578,   590,   591,   603,   606,   607,   617,   627,   636,
     645,   653,   662,   670,   678,   687,   696,   704,   712,   719,
     727,   734,   741,   747,   758,   759,   762,   763
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "lessOrEqual_token",
  "greaterOrEqual_token", "negative_infinity_token", "equivalent_token",
  "imply_token", "equal_token", "nonEqual_token",
  "positive_infinity_token", "forall_token", "exists_token", "case_token",
  "if_token", "switch_token", "then_token", "else_token",
  "otherwise_token", "sum_token", "product_token", "kronDelta_token",
  "diracDelta_token", "uniform_token", "bernoulli_token", "discrete_token",
  "normal_token", "poisson_token", "exponential_token", "weibull_token",
  "gama_token", "dirichlet_token", "multinomial_token", "types_token",
  "variables_token", "cpfs_token", "cdfs_token", "reward_token",
  "domain_token", "requirements_token", "objects_token",
  "init_state_token", "state_action_constraints_token",
  "action_preconditions_token", "state_invariants_token", "instance_token",
  "non_fluents_token", "discount_token", "terminate_when_token",
  "horizon_token", "max_nondef_actions_token", "doubleNum_token",
  "id_token", "variable_token", "enum_token", "object_token",
  "integer_token", "real_token", "bool_token", "true_token", "false_token",
  "default_token", "level_token", "observ_fluent_token",
  "action_fluent_token", "state_fluent_token", "intermediate_token",
  "derived_fluent_token", "non_fluent_token", "intNum_token",
  "AgregateOperator", "'|'", "'^'", "'&'", "'~'", "'<'", "'>'", "'+'",
  "'-'", "'*'", "'/'", "NEGATIVE", "'{'", "'}'", "'='", "';'", "','",
  "':'", "'.'", "'['", "']'", "'?'", "'('", "')'", "'$'", "'_'", "$accept",
  "Program", "RddlBlock", "DomainBlock", "DomainList",
  "RequirementsSection", "RequirementsList", "TypeSection", "TypeList",
  "SchematicType", "EnumList", "TypeSpecification", "StructMemberList",
  "LConst", "VarSection", "VarList", "VariableSchematic", "ParametarList",
  "ParametarListTypeSpecs", "RangeConstant", "StructRangeConsant",
  "StructRangeConsantList", "BoolType", "DoubleType", "IntType",
  "CPFSection", "CPFHeader", "CPFList", "CPFSchematic",
  "VariableExpression", "TermList", "Term", "MemberList", "Pterm",
  "Expression", "StructExpressionList", "ExpressionList", "TypedVarList",
  "TypedVariable", "CaseList", "CaseSchematic", "LConstCaseList",
  "RewardsSection", "StateConstraintsSection", "StateConstraintList",
  "StateConstraintSchematic", "ActionPreconditionsSection",
  "ActionPreconditionsList", "ActionPreconditionsSchematic",
  "StateInvariantSection", "StateInvariantList", "StateInvariantSchematic",
  "ObjectsSection", "ObjectsList", "ObjectsSchematic", "ObjectsConstList",
  "NonfluentBlock", "VariablesInstanceList", "VariableInstanceSchematic",
  "LConstList", "InstanceBlock", "HorizonSpecification",
  "PositiveIntOrPositiveInfinity", YY_NULLPTR
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_int16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,   293,   294,
     295,   296,   297,   298,   299,   300,   301,   302,   303,   304,
     305,   306,   307,   308,   309,   310,   311,   312,   313,   314,
     315,   316,   317,   318,   319,   320,   321,   322,   323,   324,
     325,   124,    94,    38,   126,    60,    62,    43,    45,    42,
      47,   326,   123,   125,    61,    59,    44,    58,    46,    91,
      93,    63,    40,    41,    36,    95
};
# endif

#define YYPACT_NINF (-472)

#define yypact_value_is_default(Yyn) \
  ((Yyn) == YYPACT_NINF)

#define YYTABLE_NINF (-1)

#define yytable_value_is_error(Yyn) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int16 yypact[] =
{
      52,   -30,     0,    10,   112,    52,  -472,  -472,  -472,    87,
     103,   107,  -472,  -472,  -472,  -472,   245,    44,   160,   118,
     127,  -472,  -472,    78,     9,   130,   134,   143,   155,   123,
     245,   245,   245,   245,   162,   245,   245,   245,   245,   245,
     144,   158,   198,   201,  1089,   -17,   191,   242,   777,   853,
     929,  -472,  -472,  -472,  -472,  -472,   249,  -472,  -472,  -472,
    -472,  -472,   254,   272,   233,   243,   198,   -51,   244,   201,
     235,   236,   241,   250,   255,   263,   257,   270,   271,   274,
     275,   276,   279,    22,   280,   286,   287,   288,  -472,   138,
    -472,  -472,  -472,  -472,  -472,  1089,   151,  1089,  1089,  1089,
    1089,   284,  -472,  1334,   295,   298,   277,    -2,   301,   299,
     242,   308,  1357,   311,  1089,   310,  1368,   316,  1089,   319,
    1379,   322,   320,    75,   324,   249,   300,   323,   325,   142,
     326,  -472,  -472,    71,   330,   328,  -472,   332,   336,  1089,
     -22,   341,   342,  1089,  1089,  1089,  1089,   373,  1089,  1089,
    1089,  1089,  1089,  1089,   374,   375,   -16,  1089,   -22,  -472,
    -472,  -472,  -472,  -472,  -472,  -472,  -472,   343,  -472,  -472,
    -472,   682,    80,  -472,  1089,  1089,  1089,  1089,  1089,  1089,
    1089,  1089,  1089,  1089,  1089,  1089,  1089,  1089,  1089,  -472,
     376,  -472,   344,   347,   351,   354,   352,  -472,  -472,  -472,
     353,  -472,  -472,  -472,   355,  -472,  -472,  -472,   363,  1089,
     364,  -472,  1089,   146,    39,   365,   367,   151,   385,   401,
    -472,  -472,  -472,  -472,  -472,   371,   382,    82,  -472,   411,
     411,   232,  -472,  -472,   416,  -472,   384,   411,   411,   268,
     383,   739,   394,   392,   891,   413,   705,   438,  1051,  1195,
     393,   395,  -472,  -472,  -472,   249,   249,   428,   240,  1223,
     406,   389,   415,  1089,  -472,  -472,   297,   297,  1493,   655,
     297,   297,  1502,    40,    40,   297,   297,   -33,   -33,  -472,
    -472,  -472,  -472,  -472,   417,   -29,  -472,  -472,  -472,  -472,
    -472,  -472,  1414,   421,   422,   423,   424,   437,   420,   430,
     444,  -472,  -472,   451,   441,    91,   453,   447,    71,   452,
     454,   455,   456,   457,   458,   459,   460,   463,   464,   472,
     522,  -472,   474,   475,   476,  -472,  -472,  1089,  -472,   151,
    1089,  -472,  -472,  -472,  1089,  1089,  1089,  1089,   467,   468,
    -472,    72,  -472,  1089,  -472,   240,   -22,  1241,  -472,   479,
     508,   496,  -472,   242,   -13,   528,    11,     1,   527,   242,
     -13,   497,    71,   529,   493,   500,   511,  -472,  -472,    71,
      71,    71,    71,    71,    71,   538,  1089,   411,  1089,  1089,
       6,  1089,  1089,   491,   513,   521,   516,   526,   545,   569,
    1276,  -472,  -472,  -472,  -472,  -472,  1165,   151,   -29,   540,
     530,   544,   129,   576,   546,   -13,   547,  -472,   539,  -472,
    -472,   548,   550,   552,   553,  -472,   551,   557,  -472,   585,
    -472,   559,   568,   567,   570,    38,    88,   574,  -472,  1471,
    -472,  1471,  1459,   -22,   543,   571,   575,  1471,  1471,  -472,
    1013,  -472,  -472,  -472,  -472,  -472,   151,  -472,  -472,   -29,
    -472,   580,   647,  -472,   151,   563,   581,  -472,   206,  1089,
     618,   617,   584,   586,   151,   587,   591,   588,   611,   613,
     602,   630,   608,   632,   634,  1089,   609,  1089,  -472,     6,
    -472,  1322,   607,  -472,   273,  -472,  -472,  -472,  -472,  -472,
    -472,  -472,  -472,   124,   -11,   627,   628,   652,   620,  -472,
    -472,  -472,   624,   622,   151,   131,   635,   636,   637,   638,
     604,   639,   644,   -12,   654,  -472,   653,    71,  -472,   656,
     660,  -472,   665,  -472,   666,   667,  1471,  1089,  1471,  -472,
     151,  -472,   670,   674,   679,   680,   633,   677,   643,  -472,
    -472,  -472,   124,   124,  -472,  -472,   151,   148,   673,   683,
     684,   242,   -13,     1,   721,  -472,   723,   688,   691,  -472,
    -472,  -472,   698,   647,   647,   647,   647,   647,  1471,  -472,
     -13,   -13,     1,   728,   647,   647,  -472,   703,   710,  -472,
     647,  -472,   702,     1,   743,   711,   730,   708,   736,   737,
    -472,   -13,   738,   741,   744,   749,   750,   751,   752,   755,
     754,   742,  -472,   756,   732,   747,   758,  -472,   759,   757,
     760,   762,   618,   797,   798,   767,  -472,   768,   771,   772,
     773,   774,   776,   778,   618,   811,   151,  -472,  -472,  -472,
     618,   819,    19,   199,   801,   802,   803,   804,  -472,  -472,
    -472,  -472,  -472,   219,   849,   806,   807,  -472,   808,   816,
     814,   818,   823,   825,   826,   856,   827,   831,   832,   833,
     834,   836,   869,   837,   874,   840,   -13,     1,   877,     1,
     878,   842,  -472,  -472,  -472,     1,   882,   -13,   848,  -472,
     850,  -472,   852,   854,   862,   880,   888,   886,   889,   892,
     855,   922,   924,   893,   618,   928,   618,   933,   900,   618,
     935,   902,   905,   906,   246,   907,   908,   909,   910,   913,
     912,   914,   618,   917,   918,   921,   923,   955,   926,   963,
     930,  -472,   964,   931,   932,  -472,  -472,     1,   968,   936,
    -472,   938,  -472,   942,  -472,   969,   934,   945,   979,   995,
     996,   965,   618,   997,   966,   967,   971,   999,   976,   977,
     970,   980,   985,   984,  1023,   988,  -472,  -472,  -472,   991,
     992,  -472,  -472,  1024,   993,   994,  -472
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       0,     0,     0,     0,     0,     2,     3,     5,     4,     0,
       0,     0,     1,     6,     8,     7,     0,     0,     0,     0,
       0,    94,    95,     0,     0,     0,     0,     0,     0,     0,
      25,    19,    20,    21,     0,    22,    23,    26,    27,    24,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     9,    16,    10,    11,    12,     0,    13,    14,    17,
      18,    15,     0,     0,     0,     0,    35,     0,     0,    58,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   149,    99,
     117,   119,   151,   152,   150,     0,     0,     0,     0,     0,
       0,     0,   118,     0,    32,     0,     0,     0,     0,     0,
     197,     0,     0,     0,   183,     0,     0,     0,   188,     0,
       0,     0,   193,    99,     0,    96,     0,     0,     0,     0,
       0,    36,    68,     0,     0,     0,    59,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   101,
     146,    51,    53,    56,    54,    55,    52,     0,   121,   131,
     130,     0,     0,   120,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,   180,
       0,    31,     0,     0,     0,     0,     0,   198,   182,   185,
       0,   184,   187,   190,     0,   189,   192,   195,     0,     0,
       0,    97,     0,     0,     0,     0,     0,     0,     0,     0,
      34,    45,    46,    47,    48,    70,     0,     0,    57,     0,
       0,     0,   106,   108,     0,   107,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   112,   115,   103,     0,     0,     0,   110,   168,
       0,     0,   104,     0,   142,   141,   137,   138,   147,   148,
     139,   140,   145,   143,   144,   136,   135,   129,   132,   133,
     134,    33,    30,    29,     0,     0,   196,   181,   186,   191,
     194,    93,     0,     0,     0,     0,     0,     0,     0,     0,
       0,    38,    37,     0,     0,    43,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   170,     0,
       0,   109,     0,     0,     0,   156,   158,     0,   155,     0,
       0,   162,   154,   153,     0,     0,     0,     0,     0,     0,
     116,     0,   111,     0,   122,   100,     0,     0,    28,   200,
       0,     0,    98,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    71,    69,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   114,   113,   169,   102,   105,   166,     0,     0,   202,
       0,     0,     0,     0,     0,   207,     0,   237,     0,   236,
     234,     0,     0,     0,     0,    42,    49,    43,    44,     0,
      39,     0,     0,     0,     0,     0,     0,     0,   172,   123,
     171,   124,     0,     0,     0,     0,   173,   125,   126,   159,
       0,   157,   160,   163,   164,   161,     0,   167,   201,     0,
     199,     0,     0,   210,     0,   212,     0,   208,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,   128,     0,
     178,   177,     0,   203,     0,    90,    89,    87,    75,    77,
      85,    86,    91,     0,     0,     0,     0,     0,     0,    72,
      73,    74,   215,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,    50,     0,     0,    67,     0,
       0,    65,     0,    66,     0,     0,   127,     0,   176,   174,
       0,   165,     0,     0,     0,     0,     0,     0,     0,    82,
      88,    92,     0,     0,    76,   214,     0,     0,     0,     0,
       0,     0,     0,     0,     0,   235,     0,     0,     0,   206,
     205,    40,     0,     0,     0,     0,     0,     0,   175,   179,
       0,     0,     0,     0,     0,     0,    78,     0,     0,   216,
       0,   209,     0,     0,     0,     0,     0,     0,     0,     0,
     232,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,    81,    83,     0,     0,     0,   211,     0,     0,
       0,     0,     0,     0,     0,     0,    41,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,    80,    79,   213,
       0,     0,     0,     0,     0,     0,     0,     0,    62,    60,
      63,    64,    61,     0,     0,     0,     0,    84,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,   231,   224,   204,     0,     0,     0,     0,   230,
       0,   228,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,   223,     0,     0,     0,   222,   220,     0,     0,     0,
     229,     0,   227,     0,   226,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,   221,   219,   218,     0,
       0,   225,   233,     0,     0,     0,   217
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int16 yypgoto[] =
{
    -472,  -472,  -472,  1074,   228,  -472,   -87,  -472,   867,  -472,
     717,  -345,   619,   -95,  -472,  1012,  -472,  -472,   781,  -210,
    -226,   466,  -472,  -472,  -472,  -472,  -472,   959,  -472,   -48,
     740,  -136,  -249,  -472,   -44,   696,   753,  -223,  -472,   606,
    -472,  -436,  -472,  -472,   981,  -472,  -472,  1007,  -472,  -472,
     885,  -472,  -472,  -108,  -472,  -385,  1092,  -348,  -472,  -471,
    1093,  -442,  -354
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int16 yydefgoto[] =
{
      -1,     4,     5,     6,    29,    30,   106,    31,    65,    66,
     306,   225,   303,   537,    32,    68,    69,   134,   226,   498,
     538,   539,   499,   500,   501,    33,    34,   124,   125,   102,
     261,   262,   159,   258,   112,   168,   260,   317,   318,   435,
     436,   385,    35,    36,   113,   114,    37,   117,   118,    38,
     121,   122,    39,   109,   110,   351,     7,   404,   405,   503,
       8,   298,   410
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_int16 yytable[] =
{
     103,   167,   197,   411,   236,   116,   120,   319,   126,   342,
     482,   407,   414,   448,   323,   324,   509,   416,   511,   433,
     194,   407,     9,   349,   422,   423,   424,   425,   426,   427,
     123,   232,   233,   548,   558,   104,   132,   252,   253,   402,
     540,   133,   535,   174,   175,   254,   187,   188,   178,   179,
     104,   160,    10,   169,   170,   171,   172,   457,   541,   408,
     650,   403,    11,   550,   483,   350,   105,   434,   296,   651,
     409,   559,   234,   255,   116,   579,   256,   126,   257,   299,
     409,   193,    40,   174,   175,   300,   176,   177,   178,   179,
       1,    45,   235,    46,   569,   231,   394,     2,     3,   239,
     240,   241,   242,   281,   244,   245,   246,   247,   248,   249,
     235,   150,    12,   259,   151,   183,   184,   185,   186,   187,
     188,   470,   304,   221,   471,   252,   253,   222,   223,   224,
     266,   267,   268,   269,   270,   271,   272,   273,   274,   275,
     276,   277,   278,   279,   280,   310,   311,   312,   313,   314,
     315,   180,   181,   182,   430,   183,   184,   185,   186,   187,
     188,   255,    44,   156,   256,   120,   257,   158,   292,    16,
     634,   472,   562,   265,   473,   161,   162,   363,   163,   364,
     296,   549,   645,   164,   165,    17,   293,   294,   648,    18,
     652,   654,   295,   166,   215,   296,   297,   216,    41,   587,
      42,   660,   161,   162,   586,   163,    51,   338,   339,    43,
     164,   165,    47,   452,   453,   536,    48,   217,   600,   347,
     166,   454,   598,   599,   218,    49,   156,   157,    62,   608,
     158,   219,   580,   581,   384,   174,   175,    50,   176,   177,
     178,   179,    63,   615,    56,   401,   506,   507,   296,   653,
      64,   413,   705,    67,   707,   296,   508,   710,    52,    53,
      54,    55,   716,    57,    58,    59,    60,    61,   296,   659,
     724,   174,   175,   107,   176,   177,   178,   179,    19,    20,
      21,    22,    23,   383,    24,    25,   386,    26,    27,    28,
     387,   388,   389,   390,   108,   296,   715,   476,   235,   259,
     748,   123,   167,   180,   181,   182,   127,   183,   184,   185,
     186,   187,   188,   683,   532,   685,   577,   578,   682,   533,
     129,   688,   296,   534,   128,   320,   130,   135,   341,   690,
     137,   138,   429,   139,   431,   432,   173,   437,   438,   180,
     181,   182,   140,   183,   184,   185,   186,   187,   188,   143,
     141,   384,   276,   593,   594,   595,   596,   597,   142,   502,
     192,   325,   144,   145,   602,   603,   146,   147,   148,   304,
     606,   149,   152,   736,   185,   186,   187,   188,   153,   154,
     155,   190,   196,   191,   212,   235,   174,   175,   195,   176,
     177,   178,   179,   198,   200,   202,   481,   174,   175,   204,
     176,   177,   178,   179,   206,   208,   209,   210,   213,   502,
     214,   220,   227,   228,   229,   510,   174,   175,   230,   176,
     177,   178,   179,   237,   238,   243,   250,   251,   104,   282,
     263,   526,   283,   528,   284,   384,   285,   286,   287,   305,
     288,   174,   175,   585,   176,   177,   178,   179,   289,   291,
     301,   502,   302,   307,   180,   181,   182,   308,   183,   184,
     185,   186,   187,   188,   316,   180,   181,   182,   321,   183,
     184,   185,   186,   187,   188,   309,   326,   322,   329,   336,
     340,   337,   345,   568,   180,   181,   182,   328,   183,   184,
     185,   186,   187,   188,   174,   175,   344,   176,   177,   178,
     179,   346,   348,   353,   354,   358,   331,   355,   356,   180,
     181,   182,   359,   183,   184,   185,   186,   187,   188,   174,
     175,   357,   176,   177,   178,   179,   360,   361,   362,   174,
     175,   333,   176,   177,   178,   179,   365,   366,   379,   368,
     369,   370,   371,   372,   373,   374,   376,   375,   174,   175,
     377,   176,   177,   178,   179,   378,   380,   391,   381,   382,
     399,   392,   180,   181,   182,   398,   183,   184,   185,   186,
     187,   188,   174,   175,   412,   176,   177,   178,   179,   400,
     406,   419,   415,   417,   439,   420,   421,   180,   181,   182,
     428,   183,   184,   185,   186,   187,   188,   180,   181,   182,
     440,   183,   184,   185,   186,   187,   188,   174,   175,   442,
     176,   177,   178,   179,   441,   450,   180,   181,   182,   443,
     183,   184,   185,   186,   187,   188,   449,   451,   455,   456,
     477,   459,   458,   460,   461,   462,   463,   464,   444,   465,
     180,   181,   182,   363,   183,   184,   185,   186,   187,   188,
     466,   467,   485,   468,   478,   504,   469,   486,   174,   175,
     474,   479,   445,   178,   179,   484,   505,   296,   512,   513,
     516,   514,   519,   518,   520,   180,   181,   182,   517,   183,
     184,   185,   186,   187,   188,   174,   175,   521,   176,   177,
     178,   179,   522,   523,   524,   525,   527,   555,   487,   488,
     531,   489,   542,   543,   544,   545,   490,   491,   174,   175,
     546,   176,   177,   178,   179,   547,   492,   551,   552,   576,
     574,   553,   493,   554,   556,   494,   180,   181,   182,   557,
     183,   184,   185,   186,   187,   188,   495,   560,   561,   496,
     563,   497,   174,   175,   564,   176,   177,   178,   179,   565,
     566,   567,   570,   180,   181,   182,   571,   183,   184,   185,
     186,   187,   188,   572,   575,   573,   582,   583,   588,   584,
     589,   590,   264,   591,   592,   601,   180,   181,   182,   604,
     183,   184,   185,   186,   187,   188,   605,   607,    70,    71,
     609,    72,    73,   612,   610,   332,    74,    75,    76,    77,
      78,    79,    80,    81,    82,    83,    84,    85,    86,    87,
     180,   181,   182,   611,   183,   184,   185,   186,   187,   188,
     613,   614,   627,   616,   617,   327,   625,   618,    88,    89,
      90,    91,   619,   620,   621,   622,    92,    93,   623,   624,
     628,   631,   626,   629,   630,   632,    94,   633,   635,   636,
     637,    95,    96,   638,    97,    98,   639,   640,   641,   642,
     111,   643,   646,   644,    70,    71,    99,    72,    73,   100,
     649,   101,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,   655,   656,   657,   658,
     661,   662,   663,   664,   174,   175,   666,   176,   177,   178,
     179,   665,   667,   671,    88,    89,    90,    91,   668,   669,
     672,   670,    92,    93,   673,   674,   678,   675,   677,   676,
     679,   680,    94,   681,   684,   686,   687,    95,    96,   689,
      97,    98,   691,   131,   692,   693,   115,   698,   701,   694,
      70,    71,    99,    72,    73,   100,   695,   101,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,   180,   181,   182,   696,   183,   184,   185,   186,
     187,   188,   697,   702,   699,   703,   700,   330,   704,   706,
      88,    89,    90,    91,   708,   709,   711,   712,    92,    93,
     713,   714,   717,   718,   719,   720,   721,   722,    94,   723,
     725,   726,   729,    95,    96,   727,    97,    98,   728,   730,
     731,   733,   119,   732,   734,   737,   741,   735,    99,   742,
     738,   100,   739,   101,    70,    71,   740,    72,    73,   743,
     744,   480,    74,    75,    76,    77,    78,    79,    80,    81,
      82,    83,    84,    85,    86,    87,   745,   746,   749,   747,
     753,   750,   751,   756,   174,   175,   752,   176,   177,   178,
     179,   754,   755,   757,    88,    89,    90,    91,   758,   759,
     760,   761,    92,    93,   762,   764,   763,   766,   765,    13,
     418,   136,    94,   515,   211,   529,   395,    95,    96,   367,
      97,    98,   647,   447,   290,   201,   393,    14,    15,     0,
      70,    71,    99,    72,    73,   100,     0,   101,    74,    75,
      76,    77,    78,    79,    80,    81,    82,    83,    84,    85,
      86,    87,   180,   181,   182,   205,   183,   184,   185,   186,
     187,   188,     0,     0,     0,     0,     0,   334,     0,     0,
      88,    89,    90,    91,     0,     0,     0,     0,    92,    93,
       0,     0,     0,     0,     0,     0,     0,     0,    94,     0,
       0,     0,     0,    95,    96,     0,    97,    98,     0,     0,
       0,     0,     0,     0,     0,     0,    70,    71,    99,    72,
      73,   100,     0,   101,    74,    75,    76,    77,    78,    79,
      80,    81,    82,    83,    84,    85,    86,    87,   174,   175,
       0,   176,   177,   178,   179,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,    88,    89,    90,    91,
       0,     0,     0,     0,    92,    93,   174,   175,     0,   176,
     177,   178,   179,     0,    94,     0,     0,     0,     0,    95,
       0,     0,    97,    98,   174,   175,     0,   176,   177,   178,
     179,     0,     0,     0,    99,     0,     0,   100,     0,   101,
       0,     0,     0,     0,     0,     0,   180,   181,   182,     0,
     183,   184,   185,   186,   187,   188,     0,     0,     0,   174,
     175,   335,   176,   177,   178,   179,     0,     0,     0,     0,
       0,     0,     0,     0,   180,   181,   182,     0,   183,   184,
     185,   186,   187,   188,     0,     0,     0,     0,     0,   343,
       0,     0,   180,   181,   182,     0,   183,   396,   185,   186,
     187,   188,     0,     0,     0,   174,   175,   397,   176,   177,
     178,   179,     0,     0,     0,     0,     0,   174,   175,     0,
     176,   177,   178,   179,     0,     0,     0,   180,   181,   182,
       0,   183,   184,   185,   186,   187,   188,     0,     0,     0,
     174,   175,   446,   176,   177,   178,   179,     0,     0,     0,
       0,   174,   175,     0,   176,   177,   178,   179,     0,     0,
       0,     0,   174,   175,     0,   176,   177,   178,   179,     0,
       0,     0,     0,   180,   181,   182,     0,   183,   184,   185,
     186,   187,   188,     0,     0,   180,   181,   182,   530,   183,
     184,   185,   186,   187,   188,     0,     0,   174,   175,   189,
     176,   177,   178,   179,     0,     0,     0,     0,   180,   181,
     182,     0,   183,   184,   185,   186,   187,   188,     0,   180,
     181,   182,   199,   183,   184,   185,   186,   187,   188,     0,
     180,   181,   182,   203,   183,   184,   185,   186,   187,   188,
       0,     0,   174,   175,   207,   176,   177,   178,   179,     0,
       0,     0,     0,     0,   174,   175,   475,   176,   177,   178,
     179,     0,     0,     0,     0,   180,   181,   182,     0,   183,
     184,   185,   186,   187,   188,     0,   174,   175,     0,   352,
     177,   178,   179,     0,     0,   174,   175,     0,     0,     0,
     178,   179,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     180,   181,   182,     0,   183,   184,   185,   186,   187,   188,
       0,     0,   180,   181,   182,     0,   183,   184,   185,   186,
     187,   188,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,   180,   181,   182,     0,   183,   184,
     185,   186,   187,   188,   181,   182,     0,   183,   184,   185,
     186,   187,   188
};

static const yytype_int16 yycheck[] =
{
      44,    96,   110,   357,   140,    49,    50,   230,    56,   258,
     446,    10,   360,   398,   237,   238,   458,   362,   460,    13,
     107,    10,    52,    52,   369,   370,   371,   372,   373,   374,
      52,    53,    54,   504,    46,    52,    87,    53,    54,    52,
      51,    92,   484,     3,     4,    61,    79,    80,     8,     9,
      52,    95,    52,    97,    98,    99,   100,   405,    69,    48,
      41,    74,    52,   505,   449,    94,    83,    61,    49,    50,
      69,    83,    94,    89,   118,   546,    92,   125,    94,    40,
      69,    83,    38,     3,     4,    46,     6,     7,     8,     9,
      38,    82,   140,    84,   530,   139,   345,    45,    46,   143,
     144,   145,   146,   190,   148,   149,   150,   151,   152,   153,
     158,    89,     0,   157,    92,    75,    76,    77,    78,    79,
      80,    83,   217,    52,    86,    53,    54,    56,    57,    58,
     174,   175,   176,   177,   178,   179,   180,   181,   182,   183,
     184,   185,   186,   187,   188,    63,    64,    65,    66,    67,
      68,    71,    72,    73,   377,    75,    76,    77,    78,    79,
      80,    89,    84,    88,    92,   209,    94,    92,   212,    82,
     612,    83,   517,    93,    86,    51,    52,    86,    54,    88,
      49,    50,   624,    59,    60,    82,    40,    41,   630,    82,
     632,   633,    46,    69,    52,    49,    50,    55,    38,   553,
      82,   643,    51,    52,   552,    54,    83,   255,   256,    82,
      59,    60,    82,    84,    85,    91,    82,    75,   572,   263,
      69,    92,   570,   571,    82,    82,    88,    89,    84,   583,
      92,    89,    84,    85,   329,     3,     4,    82,     6,     7,
       8,     9,    84,   591,    82,   353,    40,    41,    49,    50,
      52,   359,   694,    52,   696,    49,    50,   699,    30,    31,
      32,    33,   704,    35,    36,    37,    38,    39,    49,    50,
     712,     3,     4,    82,     6,     7,     8,     9,    33,    34,
      35,    36,    37,   327,    39,    40,   330,    42,    43,    44,
     334,   335,   336,   337,    52,    49,    50,   433,   346,   343,
     742,    52,   397,    71,    72,    73,    52,    75,    76,    77,
      78,    79,    80,   667,    41,   669,   542,   543,   666,    46,
      87,   675,    49,    50,    52,    93,    83,    83,    88,   677,
      95,    95,   376,    92,   378,   379,    52,   381,   382,    71,
      72,    73,    92,    75,    76,    77,    78,    79,    80,    92,
      95,   446,   396,   563,   564,   565,   566,   567,    95,   454,
      83,    93,    92,    92,   574,   575,    92,    92,    92,   464,
     580,    92,    92,   727,    77,    78,    79,    80,    92,    92,
      92,    86,    83,    85,    84,   433,     3,     4,    87,     6,
       7,     8,     9,    85,    83,    85,   440,     3,     4,    83,
       6,     7,     8,     9,    85,    83,    86,    83,    85,   504,
      85,    85,    82,    85,    82,   459,     3,     4,    82,     6,
       7,     8,     9,    82,    82,    52,    52,    52,    52,    85,
      87,   475,    85,   477,    83,   530,    82,    85,    85,    54,
      85,     3,     4,   551,     6,     7,     8,     9,    85,    85,
      85,   546,    85,    52,    71,    72,    73,    86,    75,    76,
      77,    78,    79,    80,    53,    71,    72,    73,    52,    75,
      76,    77,    78,    79,    80,    93,    93,    93,    86,    86,
      52,    86,    93,   527,    71,    72,    73,    93,    75,    76,
      77,    78,    79,    80,     3,     4,    90,     6,     7,     8,
       9,    86,    85,    82,    82,    85,    93,    84,    84,    71,
      72,    73,    82,    75,    76,    77,    78,    79,    80,     3,
       4,    84,     6,     7,     8,     9,    82,    76,    87,     3,
       4,    93,     6,     7,     8,     9,    83,    90,    16,    87,
      86,    86,    86,    86,    86,    86,    83,    87,     3,     4,
      86,     6,     7,     8,     9,    83,    82,    90,    83,    83,
      52,    93,    71,    72,    73,    86,    75,    76,    77,    78,
      79,    80,     3,     4,    47,     6,     7,     8,     9,    83,
      52,    88,    85,    54,    93,    85,    75,    71,    72,    73,
      52,    75,    76,    77,    78,    79,    80,    71,    72,    73,
      87,    75,    76,    77,    78,    79,    80,     3,     4,    93,
       6,     7,     8,     9,    93,    85,    71,    72,    73,    93,
      75,    76,    77,    78,    79,    80,    86,    83,    52,    83,
      87,    92,    85,    85,    84,    83,    83,    86,    93,    54,
      71,    72,    73,    86,    75,    76,    77,    78,    79,    80,
      91,    83,     5,    86,    83,    92,    86,    10,     3,     4,
      86,    86,    93,     8,     9,    85,    85,    49,    51,    85,
      83,    85,    61,    85,    61,    71,    72,    73,    87,    75,
      76,    77,    78,    79,    80,     3,     4,    85,     6,     7,
       8,     9,    62,    85,    62,    61,    87,    93,    51,    52,
      93,    54,    75,    75,    52,    85,    59,    60,     3,     4,
      86,     6,     7,     8,     9,    93,    69,    82,    82,    76,
      87,    84,    75,    85,    85,    78,    71,    72,    73,    85,
      75,    76,    77,    78,    79,    80,    89,    83,    85,    92,
      84,    94,     3,     4,    84,     6,     7,     8,     9,    84,
      84,    84,    82,    71,    72,    73,    82,    75,    76,    77,
      78,    79,    80,    84,    87,    85,    93,    84,    47,    85,
      47,    83,    90,    82,    76,    47,    71,    72,    73,    76,
      75,    76,    77,    78,    79,    80,    76,    85,    11,    12,
      47,    14,    15,    85,    83,    90,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    29,    30,    31,    32,
      71,    72,    73,    83,    75,    76,    77,    78,    79,    80,
      84,    84,    90,    85,    83,    86,    84,    83,    51,    52,
      53,    54,    83,    83,    83,    83,    59,    60,    83,    85,
      93,    84,    86,    85,    85,    85,    69,    85,    51,    51,
      83,    74,    75,    85,    77,    78,    85,    85,    85,    85,
      83,    85,    51,    85,    11,    12,    89,    14,    15,    92,
      51,    94,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    85,    85,    85,    85,
      41,    85,    85,    85,     3,     4,    82,     6,     7,     8,
       9,    85,    84,    47,    51,    52,    53,    54,    85,    84,
      83,    85,    59,    60,    83,    83,    47,    84,    82,    85,
      83,    47,    69,    83,    47,    47,    84,    74,    75,    47,
      77,    78,    84,    66,    84,    83,    83,    51,    83,    85,
      11,    12,    89,    14,    15,    92,    84,    94,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    71,    72,    73,    85,    75,    76,    77,    78,
      79,    80,    84,    51,    85,    51,    84,    86,    85,    51,
      51,    52,    53,    54,    51,    85,    51,    85,    59,    60,
      85,    85,    85,    85,    85,    85,    83,    85,    69,    85,
      83,    83,    47,    74,    75,    84,    77,    78,    85,    83,
      47,    47,    83,    83,    83,    47,    47,    85,    89,    85,
      84,    92,    84,    94,    11,    12,    84,    14,    15,    84,
      51,    18,    19,    20,    21,    22,    23,    24,    25,    26,
      27,    28,    29,    30,    31,    32,    51,    51,    51,    84,
      51,    85,    85,    83,     3,     4,    85,     6,     7,     8,
       9,    85,    85,    83,    51,    52,    53,    54,    83,    85,
      47,    83,    59,    60,    83,    51,    84,    83,    85,     5,
     363,    69,    69,   464,   125,   479,   346,    74,    75,   308,
      77,    78,   626,   397,   209,   114,   343,     5,     5,    -1,
      11,    12,    89,    14,    15,    92,    -1,    94,    19,    20,
      21,    22,    23,    24,    25,    26,    27,    28,    29,    30,
      31,    32,    71,    72,    73,   118,    75,    76,    77,    78,
      79,    80,    -1,    -1,    -1,    -1,    -1,    86,    -1,    -1,
      51,    52,    53,    54,    -1,    -1,    -1,    -1,    59,    60,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    69,    -1,
      -1,    -1,    -1,    74,    75,    -1,    77,    78,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    11,    12,    89,    14,
      15,    92,    -1,    94,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,     3,     4,
      -1,     6,     7,     8,     9,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    51,    52,    53,    54,
      -1,    -1,    -1,    -1,    59,    60,     3,     4,    -1,     6,
       7,     8,     9,    -1,    69,    -1,    -1,    -1,    -1,    74,
      -1,    -1,    77,    78,     3,     4,    -1,     6,     7,     8,
       9,    -1,    -1,    -1,    89,    -1,    -1,    92,    -1,    94,
      -1,    -1,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,
      75,    76,    77,    78,    79,    80,    -1,    -1,    -1,     3,
       4,    86,     6,     7,     8,     9,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    75,    76,
      77,    78,    79,    80,    -1,    -1,    -1,    -1,    -1,    86,
      -1,    -1,    71,    72,    73,    -1,    75,    76,    77,    78,
      79,    80,    -1,    -1,    -1,     3,     4,    86,     6,     7,
       8,     9,    -1,    -1,    -1,    -1,    -1,     3,     4,    -1,
       6,     7,     8,     9,    -1,    -1,    -1,    71,    72,    73,
      -1,    75,    76,    77,    78,    79,    80,    -1,    -1,    -1,
       3,     4,    86,     6,     7,     8,     9,    -1,    -1,    -1,
      -1,     3,     4,    -1,     6,     7,     8,     9,    -1,    -1,
      -1,    -1,     3,     4,    -1,     6,     7,     8,     9,    -1,
      -1,    -1,    -1,    71,    72,    73,    -1,    75,    76,    77,
      78,    79,    80,    -1,    -1,    71,    72,    73,    86,    75,
      76,    77,    78,    79,    80,    -1,    -1,     3,     4,    85,
       6,     7,     8,     9,    -1,    -1,    -1,    -1,    71,    72,
      73,    -1,    75,    76,    77,    78,    79,    80,    -1,    71,
      72,    73,    85,    75,    76,    77,    78,    79,    80,    -1,
      71,    72,    73,    85,    75,    76,    77,    78,    79,    80,
      -1,    -1,     3,     4,    85,     6,     7,     8,     9,    -1,
      -1,    -1,    -1,    -1,     3,     4,    17,     6,     7,     8,
       9,    -1,    -1,    -1,    -1,    71,    72,    73,    -1,    75,
      76,    77,    78,    79,    80,    -1,     3,     4,    -1,    85,
       7,     8,     9,    -1,    -1,     3,     4,    -1,    -1,    -1,
       8,     9,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      71,    72,    73,    -1,    75,    76,    77,    78,    79,    80,
      -1,    -1,    71,    72,    73,    -1,    75,    76,    77,    78,
      79,    80,    -1,    -1,    -1,    -1,    -1,    -1,    -1,    -1,
      -1,    -1,    -1,    -1,    71,    72,    73,    -1,    75,    76,
      77,    78,    79,    80,    72,    73,    -1,    75,    76,    77,
      78,    79,    80
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    38,    45,    46,    97,    98,    99,   152,   156,    52,
      52,    52,     0,    99,   152,   156,    82,    82,    82,    33,
      34,    35,    36,    37,    39,    40,    42,    43,    44,   100,
     101,   103,   110,   121,   122,   138,   139,   142,   145,   148,
      38,    38,    82,    82,    84,    82,    84,    82,    82,    82,
      82,    83,   100,   100,   100,   100,    82,   100,   100,   100,
     100,   100,    84,    84,    52,   104,   105,    52,   111,   112,
      11,    12,    14,    15,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    51,    52,
      53,    54,    59,    60,    69,    74,    75,    77,    78,    89,
      92,    94,   125,   130,    52,    83,   102,    82,    52,   149,
     150,    83,   130,   140,   141,    83,   130,   143,   144,    83,
     130,   146,   147,    52,   123,   124,   125,    52,    52,    87,
      83,   104,    87,    92,   113,    83,   111,    95,    95,    92,
      92,    95,    95,    92,    92,    92,    92,    92,    92,    92,
      89,    92,    92,    92,    92,    92,    88,    89,    92,   128,
     130,    51,    52,    54,    59,    60,    69,   109,   131,   130,
     130,   130,   130,    52,     3,     4,     6,     7,     8,     9,
      71,    72,    73,    75,    76,    77,    78,    79,    80,    85,
      86,    85,    83,    83,   102,    87,    83,   149,    85,    85,
      83,   140,    85,    85,    83,   143,    85,    85,    83,    86,
      83,   123,    84,    85,    85,    52,    55,    75,    82,    89,
      85,    52,    56,    57,    58,   107,   114,    82,    85,    82,
      82,   130,    53,    54,    94,   125,   127,    82,    82,   130,
     130,   130,   130,    52,   130,   130,   130,   130,   130,   130,
      52,    52,    53,    54,    61,    89,    92,    94,   129,   130,
     132,   126,   127,    87,    90,    93,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   102,    85,    85,    83,    82,    85,    85,    85,    85,
     146,    85,   130,    40,    41,    46,    49,    50,   157,    40,
      46,    85,    85,   108,   109,    54,   106,    52,    86,    93,
      63,    64,    65,    66,    67,    68,    53,   133,   134,   133,
      93,    52,    93,   133,   133,    93,    93,    86,    93,    86,
      86,    93,    90,    93,    86,    86,    86,    86,   125,   125,
      52,    88,   128,    86,    90,    93,    86,   130,    85,    52,
      94,   151,    85,    82,    82,    84,    84,    84,    85,    82,
      82,    76,    87,    86,    88,    83,    90,   114,    87,    86,
      86,    86,    86,    86,    86,    87,    83,    86,    83,    16,
      82,    83,    83,   130,   109,   137,   130,   130,   130,   130,
     130,    90,    93,   132,   128,   126,    76,    86,    86,    52,
      83,   149,    52,    74,   153,   154,    52,    10,    48,    69,
     158,   158,    47,   149,   153,    85,   107,    54,   106,    88,
      85,    75,   107,   107,   107,   107,   107,   107,    52,   130,
     133,   130,   130,    13,    61,   135,   136,   130,   130,    93,
      87,    93,    93,    93,    93,    93,    86,   131,   151,    86,
      85,    83,    84,    85,    92,    52,    83,   153,    85,    92,
      85,    84,    83,    83,    86,    54,    91,    83,    86,    86,
      83,    86,    83,    86,    86,    17,   127,    87,    83,    86,
      18,   130,   137,   151,    85,     5,    10,    51,    52,    54,
      59,    60,    69,    75,    78,    89,    92,    94,   115,   118,
     119,   120,   109,   155,    92,    85,    40,    41,    50,   157,
     130,   157,    51,    85,    85,   108,    83,    87,    85,    61,
      61,    85,    62,    85,    62,    61,   130,    87,   130,   135,
      86,    93,    41,    46,    50,   157,    91,   109,   116,   117,
      51,    69,    75,    75,    52,    85,    86,    93,   155,    50,
     157,    82,    82,    84,    85,    93,    85,    85,    46,    83,
      83,    85,   107,    84,    84,    84,    84,    84,   130,   137,
      82,    82,    84,    85,    87,    87,    76,   116,   116,   155,
      84,    85,    93,    84,    85,   149,   153,   158,    47,    47,
      83,    82,    76,   115,   115,   115,   115,   115,   153,   153,
     158,    47,   115,   115,    76,    76,   115,    85,   158,    47,
      83,    83,    85,    84,    84,   153,    85,    83,    83,    83,
      83,    83,    83,    83,    85,    84,    86,    90,    93,    85,
      85,    84,    85,    85,   157,    51,    51,    83,    85,    85,
      85,    85,    85,    85,    85,   157,    51,   117,   157,    51,
      41,    50,   157,    50,   157,    85,    85,    85,    85,    50,
     157,    41,    85,    85,    85,    85,    82,    84,    85,    84,
      85,    47,    83,    83,    83,    84,    85,    82,    47,    83,
      47,    83,   153,   158,    47,   158,    47,    84,   158,    47,
     153,    84,    84,    83,    85,    84,    85,    84,    51,    85,
      84,    83,    51,    51,    85,   157,    51,   157,    51,    85,
     157,    51,    85,    85,    85,    50,   157,    85,    85,    85,
      85,    83,    85,    85,   157,    83,    83,    84,    85,    47,
      83,    47,    83,    47,    83,    85,   158,    47,    84,    84,
      84,    47,    85,    84,    51,    51,    51,    84,   157,    51,
      85,    85,    85,    51,    85,    85,    83,    83,    83,    85,
      47,    83,    83,    84,    51,    85,    83
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    96,    97,    98,    98,    98,    98,    98,    98,    99,
     100,   100,   100,   100,   100,   100,   100,   100,   100,   100,
     100,   100,   100,   100,   100,   100,   100,   100,   101,   101,
     101,   101,   102,   102,   103,   104,   104,   105,   105,   105,
     105,   105,   105,   106,   106,   107,   107,   107,   107,   108,
     108,   109,   109,   109,   109,   109,   109,   110,   111,   111,
     112,   112,   112,   112,   112,   112,   112,   112,   113,   113,
     114,   114,   115,   115,   115,   115,   115,   115,   115,   115,
     115,   116,   116,   117,   117,   118,   118,   119,   119,   119,
     119,   120,   120,   121,   122,   122,   123,   123,   124,   125,
     125,   125,   125,   125,   126,   126,   127,   127,   127,   127,
     128,   128,   129,   129,   129,   129,   129,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   130,   130,   130,   130,
     130,   130,   130,   130,   130,   130,   131,   131,   132,   132,
     133,   133,   134,   135,   135,   136,   136,   137,   137,   137,
     138,   139,   139,   140,   140,   141,   142,   142,   143,   143,
     144,   145,   145,   146,   146,   147,   148,   149,   149,   150,
     151,   151,   151,   151,   152,   152,   152,   153,   153,   154,
     154,   154,   154,   154,   154,   155,   155,   156,   156,   156,
     156,   156,   156,   156,   156,   156,   156,   156,   156,   156,
     156,   156,   156,   156,   157,   157,   158,   158
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_int8 yyr2[] =
{
       0,     2,     1,     1,     1,     1,     2,     2,     2,     5,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     1,
       1,     1,     1,     1,     1,     1,     1,     1,     6,     5,
       5,     4,     1,     3,     5,     1,     2,     4,     4,     6,
       9,    11,     6,     1,     3,     1,     1,     1,     1,     3,
       5,     1,     1,     1,     1,     1,     1,     5,     1,     2,
      12,    12,    12,    12,    12,     8,     8,     8,     1,     4,
       1,     3,     1,     1,     1,     1,     2,     1,     3,     5,
       5,     3,     1,     3,     5,     1,     1,     1,     2,     1,
       1,     1,     2,     5,     1,     1,     1,     2,     4,     1,
       4,     2,     5,     3,     1,     3,     1,     1,     1,     2,
       2,     3,     1,     3,     3,     1,     2,     1,     1,     1,
       2,     2,     4,     6,     6,     6,     6,     8,     7,     3,
       2,     2,     3,     3,     3,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     3,     3,     2,     3,     3,     1,
       1,     1,     1,     4,     4,     4,     4,     6,     4,     6,
       6,     6,     4,     6,     6,     8,     4,     5,     1,     3,
       1,     3,     3,     1,     3,     4,     3,     3,     3,     5,
       4,     5,     4,     1,     2,     2,     5,     4,     1,     2,
       2,     5,     4,     1,     3,     2,     5,     1,     2,     6,
       1,     3,     2,     4,    18,    13,    13,     1,     2,     5,
       2,     6,     2,     7,     4,     1,     3,    32,    28,    27,
      23,    27,    23,    22,    18,    28,    24,    23,    19,    23,
      19,    18,    14,    29,     3,     6,     1,     1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                    \
  do                                                              \
    if (yychar == YYEMPTY)                                        \
      {                                                           \
        yychar = (Token);                                         \
        yylval = (Value);                                         \
        YYPOPSTACK (yylen);                                       \
        yystate = *yyssp;                                         \
        goto yybackup;                                            \
      }                                                           \
    else                                                          \
      {                                                           \
        yyerror (YY_("syntax error: cannot back up")); \
        YYERROR;                                                  \
      }                                                           \
  while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*-----------------------------------.
| Print this symbol's value on YYO.  |
`-----------------------------------*/

static void
yy_symbol_value_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyoutput = yyo;
  YYUSE (yyoutput);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyo, yytoknum[yytype], *yyvaluep);
# endif
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}


/*---------------------------.
| Print this symbol on YYO.  |
`---------------------------*/

static void
yy_symbol_print (FILE *yyo, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyo, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyo, yytype, yyvaluep);
  YYFPRINTF (yyo, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yy_state_t *yybottom, yy_state_t *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yy_state_t *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %d):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[+yyssp[yyi + 1 - yynrhs]],
                       &yyvsp[(yyi + 1) - (yynrhs)]
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen(S) (YY_CAST (YYPTRDIFF_T, strlen (S)))
#  else
/* Return the length of YYSTR.  */
static YYPTRDIFF_T
yystrlen (const char *yystr)
{
  YYPTRDIFF_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYPTRDIFF_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYPTRDIFF_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            else
              goto append;

          append:
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (yyres)
    return yystpcpy (yyres, yystr) - yyres;
  else
    return yystrlen (yystr);
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYPTRDIFF_T *yymsg_alloc, char **yymsg,
                yy_state_t *yyssp, int yytoken)
{
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULLPTR;
  /* Arguments of yyformat: reported tokens (one for the "unexpected",
     one per "expected"). */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Actual size of YYARG. */
  int yycount = 0;
  /* Cumulated lengths of YYARG.  */
  YYPTRDIFF_T yysize = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[+*yyssp];
      YYPTRDIFF_T yysize0 = yytnamerr (YY_NULLPTR, yytname[yytoken]);
      yysize = yysize0;
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYPTRDIFF_T yysize1
                    = yysize + yytnamerr (YY_NULLPTR, yytname[yyx]);
                  if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
                    yysize = yysize1;
                  else
                    return 2;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
    default: /* Avoid compiler warnings. */
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    /* Don't count the "%s"s in the final size, but reserve room for
       the terminator.  */
    YYPTRDIFF_T yysize1 = yysize + (yystrlen (yyformat) - 2 * yycount) + 1;
    if (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM)
      yysize = yysize1;
    else
      return 2;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          ++yyp;
          ++yyformat;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    yy_state_fast_t yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yy_state_t yyssa[YYINITDEPTH];
    yy_state_t *yyss;
    yy_state_t *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYPTRDIFF_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYPTRDIFF_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;


/*------------------------------------------------------------.
| yynewstate -- push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;


/*--------------------------------------------------------------------.
| yysetstate -- set current state (the top of the stack) to yystate.  |
`--------------------------------------------------------------------*/
yysetstate:
  YYDPRINTF ((stderr, "Entering state %d\n", yystate));
  YY_ASSERT (0 <= yystate && yystate < YYNSTATES);
  YY_IGNORE_USELESS_CAST_BEGIN
  *yyssp = YY_CAST (yy_state_t, yystate);
  YY_IGNORE_USELESS_CAST_END

  if (yyss + yystacksize - 1 <= yyssp)
#if !defined yyoverflow && !defined YYSTACK_RELOCATE
    goto yyexhaustedlab;
#else
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYPTRDIFF_T yysize = yyssp - yyss + 1;

# if defined yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        yy_state_t *yyss1 = yyss;
        YYSTYPE *yyvs1 = yyvs;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * YYSIZEOF (*yyssp),
                    &yyvs1, yysize * YYSIZEOF (*yyvsp),
                    &yystacksize);
        yyss = yyss1;
        yyvs = yyvs1;
      }
# else /* defined YYSTACK_RELOCATE */
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yy_state_t *yyss1 = yyss;
        union yyalloc *yyptr =
          YY_CAST (union yyalloc *,
                   YYSTACK_ALLOC (YY_CAST (YYSIZE_T, YYSTACK_BYTES (yystacksize))));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
# undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YY_IGNORE_USELESS_CAST_BEGIN
      YYDPRINTF ((stderr, "Stack size increased to %ld\n",
                  YY_CAST (long, yystacksize)));
      YY_IGNORE_USELESS_CAST_END

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }
#endif /* !defined yyoverflow && !defined YYSTACK_RELOCATE */

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;


/*-----------.
| yybackup.  |
`-----------*/
yybackup:
  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);
  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  /* Discard the shifted token.  */
  yychar = YYEMPTY;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
  case 2:
#line 101 "parser.ypp"
                              { }
#line 2088 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 3:
#line 107 "parser.ypp"
                                    { (yyval.rddlTask) = rddlTask; }
#line 2094 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 4:
#line 108 "parser.ypp"
                                    { (yyval.rddlTask) = rddlTask; }
#line 2100 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 5:
#line 109 "parser.ypp"
                                    { (yyval.rddlTask) = rddlTask; }
#line 2106 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 6:
#line 110 "parser.ypp"
                                    { (yyval.rddlTask) = (yyvsp[-1].rddlTask); }
#line 2112 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 7:
#line 111 "parser.ypp"
                                    { (yyval.rddlTask) = (yyvsp[-1].rddlTask); }
#line 2118 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 8:
#line 112 "parser.ypp"
                                    { (yyval.rddlTask) = (yyvsp[-1].rddlTask); }
#line 2124 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 9:
#line 120 "parser.ypp"
                                                          {  rddlTask->domainName = *(yyvsp[-3].str); }
#line 2130 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 10:
#line 126 "parser.ypp"
                                                        { }
#line 2136 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 11:
#line 127 "parser.ypp"
                                                        { }
#line 2142 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 12:
#line 128 "parser.ypp"
                                                        { }
#line 2148 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 13:
#line 129 "parser.ypp"
                                                        { }
#line 2154 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 14:
#line 130 "parser.ypp"
                                                        { }
#line 2160 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 15:
#line 131 "parser.ypp"
                                                        { }
#line 2166 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 16:
#line 132 "parser.ypp"
                                                        { }
#line 2172 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 17:
#line 133 "parser.ypp"
                                                        { }
#line 2178 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 18:
#line 134 "parser.ypp"
                                                        { }
#line 2184 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 19:
#line 135 "parser.ypp"
                                              { }
#line 2190 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 20:
#line 136 "parser.ypp"
                                              { }
#line 2196 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 21:
#line 137 "parser.ypp"
                                              { }
#line 2202 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 22:
#line 138 "parser.ypp"
                                              { }
#line 2208 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 23:
#line 139 "parser.ypp"
                                              { }
#line 2214 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 24:
#line 140 "parser.ypp"
                                              { }
#line 2220 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 25:
#line 141 "parser.ypp"
                                              { }
#line 2226 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 26:
#line 142 "parser.ypp"
                                              { }
#line 2232 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 27:
#line 143 "parser.ypp"
                                              { }
#line 2238 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 28:
#line 148 "parser.ypp"
                                                                         { }
#line 2244 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 29:
#line 149 "parser.ypp"
                                                                         { }
#line 2250 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 30:
#line 150 "parser.ypp"
                                                                         { }
#line 2256 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 31:
#line 151 "parser.ypp"
                                                                         { }
#line 2262 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 32:
#line 154 "parser.ypp"
                                              { }
#line 2268 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 33:
#line 155 "parser.ypp"
                                              { }
#line 2274 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 34:
#line 161 "parser.ypp"
                                              { }
#line 2280 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 35:
#line 164 "parser.ypp"
                                    { }
#line 2286 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 36:
#line 165 "parser.ypp"
                                    { }
#line 2292 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 37:
#line 169 "parser.ypp"
                                                                                { rddlTask->addType(*(yyvsp[-3].str), *(yyvsp[-1].str)); }
#line 2298 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 38:
#line 170 "parser.ypp"
                                                                                { rddlTask->addType(*(yyvsp[-3].str), *(yyvsp[-1].str)); }
#line 2304 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 39:
#line 171 "parser.ypp"
                                                                                { rddlTask->addType(*(yyvsp[-5].str));
                                                                                  for (const std::string& s : *(yyvsp[-2].strs)) {
                                                                                      rddlTask->addObject(*(yyvsp[-5].str), s);
                                                                                  }}
#line 2313 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 40:
#line 175 "parser.ypp"
                                                                                { SystemUtils::abort("Defining types using Enum range error. Not implemented yet.") ; }
#line 2319 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 41:
#line 176 "parser.ypp"
                                                                                { SystemUtils::abort("Defining types using TypeSpecification error. Not implemented yet.") ;}
#line 2325 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 42:
#line 177 "parser.ypp"
                                                                                { SystemUtils::abort("Defining types using StructMemberList error. Not implemented yet.") ;}
#line 2331 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 43:
#line 180 "parser.ypp"
                                    { (yyval.strs) = new std::vector<std::string>(); (yyval.strs)->push_back(*(yyvsp[0].str)); }
#line 2337 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 44:
#line 181 "parser.ypp"
                                    { (yyval.strs) = (yyvsp[0].strs); (yyval.strs)->insert((yyval.strs)->begin(), *(yyvsp[-2].str)); }
#line 2343 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 45:
#line 184 "parser.ypp"
                                 { (yyval.type) = rddlTask->getType(*(yyvsp[0].str)); }
#line 2349 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 46:
#line 185 "parser.ypp"
                                 { (yyval.type) = rddlTask->getType(*(yyvsp[0].str)); }
#line 2355 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 47:
#line 186 "parser.ypp"
                                 { (yyval.type) = rddlTask->getType(*(yyvsp[0].str)); }
#line 2361 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 48:
#line 187 "parser.ypp"
                                 { (yyval.type) = rddlTask->getType(*(yyvsp[0].str)); }
#line 2367 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 49:
#line 190 "parser.ypp"
                                                                    { SystemUtils::abort("StructMemberList not implemented yet."); }
#line 2373 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 50:
#line 191 "parser.ypp"
                                                                    { SystemUtils::abort("StructMemberList not implemented yet."); }
#line 2379 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 51:
#line 194 "parser.ypp"
                        { (yyval.logicalExpression) = new NumericConstant((yyvsp[0].d));  }
#line 2385 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 52:
#line 195 "parser.ypp"
                        { (yyval.logicalExpression) = new NumericConstant((yyvsp[0].i));  }
#line 2391 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 53:
#line 196 "parser.ypp"
                        { (yyval.logicalExpression) = new Parameter(*(yyvsp[0].str));       }
#line 2397 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 54:
#line 197 "parser.ypp"
                        { (yyval.logicalExpression) = new NumericConstant(1.0); }
#line 2403 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 55:
#line 198 "parser.ypp"
                        { (yyval.logicalExpression) = new NumericConstant(0.0); }
#line 2409 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 56:
#line 199 "parser.ypp"
                        { assert(rddlTask->objects.find(*(yyvsp[0].str)) != rddlTask->objects.end()); (yyval.logicalExpression) = rddlTask->objects[*(yyvsp[0].str)]; }
#line 2415 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 57:
#line 205 "parser.ypp"
                                                { (yyval.parametrizedVariables) = (yyvsp[-2].parametrizedVariables); }
#line 2421 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 58:
#line 208 "parser.ypp"
                                       { (yyval.parametrizedVariables) = new std::vector<ParametrizedVariable*>(); (yyval.parametrizedVariables)->push_back((yyvsp[0].parametrizedVariable)); }
#line 2427 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 59:
#line 209 "parser.ypp"
                                       { (yyval.parametrizedVariables) = (yyvsp[0].parametrizedVariables); (yyval.parametrizedVariables)->insert((yyval.parametrizedVariables)->begin(), (yyvsp[-1].parametrizedVariable)); }
#line 2433 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 60:
#line 212 "parser.ypp"
                                                                                                                                       { (yyval.parametrizedVariable) = new ParametrizedVariable(*(yyvsp[-11].str), *(yyvsp[-10].parameters), ParametrizedVariable::STATE_FLUENT, (yyvsp[-6].type), (yyvsp[-2].d)); rddlTask->addVariableSchematic((yyval.parametrizedVariable)); }
#line 2439 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 61:
#line 213 "parser.ypp"
                                                                                                                                       { (yyval.parametrizedVariable) = new ParametrizedVariable(*(yyvsp[-11].str), *(yyvsp[-10].parameters), ParametrizedVariable::NON_FLUENT, (yyvsp[-6].type), (yyvsp[-2].d)); rddlTask->addVariableSchematic((yyval.parametrizedVariable)); }
#line 2445 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 62:
#line 214 "parser.ypp"
                                                                                                                                       { (yyval.parametrizedVariable) = new ParametrizedVariable(*(yyvsp[-11].str), *(yyvsp[-10].parameters), ParametrizedVariable::ACTION_FLUENT, (yyvsp[-6].type),  (yyvsp[-2].d)); rddlTask->addVariableSchematic((yyval.parametrizedVariable)); }
#line 2451 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 63:
#line 215 "parser.ypp"
                                                                                                                                       { SystemUtils::abort("interm-fluent parametrized variables definition not implemented. "); }
#line 2457 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 64:
#line 216 "parser.ypp"
                                                                                                                                       { SystemUtils::abort("derived-fluent parametrized variables definition not implemented. "); }
#line 2463 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 65:
#line 217 "parser.ypp"
                                                                                                                                       { SystemUtils::abort("interm-fluent parametrized variables definition not implemented. "); }
#line 2469 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 66:
#line 218 "parser.ypp"
                                                                                                                                       { SystemUtils::abort("derived-fluent parametrized variables definition not implemented. "); }
#line 2475 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 67:
#line 219 "parser.ypp"
                                                                                                                                       { SystemUtils::abort("observ-fluent parametrized variables definition not implemented. "); }
#line 2481 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 68:
#line 222 "parser.ypp"
                                                  { (yyval.parameters) = new std::vector<Parameter*>(); }
#line 2487 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 69:
#line 223 "parser.ypp"
                                                  { (yyval.parameters) = (yyvsp[-2].parameters); }
#line 2493 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 70:
#line 226 "parser.ypp"
                                                                         { (yyval.parameters) = new std::vector<Parameter*>(); (yyval.parameters)->push_back(new Parameter((yyvsp[0].type)->name, (yyvsp[0].type))); }
#line 2499 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 71:
#line 227 "parser.ypp"
                                                                         { (yyval.parameters) = (yyvsp[0].parameters); (yyval.parameters)->insert((yyval.parameters)->begin(), new Parameter((yyvsp[-2].type)->name, (yyvsp[-2].type))); }
#line 2505 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 72:
#line 230 "parser.ypp"
                                                     { (yyval.d) = (*(yyvsp[0].str) == "true") ? 1 : 0; }
#line 2511 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 73:
#line 231 "parser.ypp"
                                                     { (yyval.d) = (yyvsp[0].d); }
#line 2517 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 74:
#line 232 "parser.ypp"
                                                     { (yyval.d) = (yyvsp[0].d); }
#line 2523 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 75:
#line 233 "parser.ypp"
                                                     { Object* obj = rddlTask->getObject(*(yyvsp[0].str)); (yyval.d) = obj->value; }
#line 2529 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 76:
#line 234 "parser.ypp"
                                                     { SystemUtils::abort("Range constant variable with tag '$' definition implemented yet."); }
#line 2535 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 77:
#line 235 "parser.ypp"
                                                     { assert(rddlTask->objects.find(*(yyvsp[0].str)) != rddlTask->objects.end()); (yyval.d) = rddlTask->objects[*(yyvsp[0].str)]->value; }
#line 2541 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 78:
#line 236 "parser.ypp"
                                                     { SystemUtils::abort("< Structured ranged constatnts > not implemented yet."); }
#line 2547 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 79:
#line 237 "parser.ypp"
                                                     { SystemUtils::abort("(< Structured ranged constatnts >) not implemented yet."); }
#line 2553 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 80:
#line 238 "parser.ypp"
                                                     { SystemUtils::abort("[< Structured ranged constatnts >] not implemented yet."); }
#line 2559 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 81:
#line 242 "parser.ypp"
                                           { SystemUtils::abort("StructRangeConsant not implemented yet."); }
#line 2565 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 82:
#line 243 "parser.ypp"
                                           { SystemUtils::abort("StructRangeConsant not implemented yet."); }
#line 2571 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 83:
#line 246 "parser.ypp"
                                                                           { SystemUtils::abort("StructRangeConsantList not implemented yet."); }
#line 2577 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 84:
#line 247 "parser.ypp"
                                                                           { SystemUtils::abort("StructRangeConsantList not implemented yet."); }
#line 2583 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 85:
#line 250 "parser.ypp"
                      { (yyval.str) = (yyvsp[0].str); }
#line 2589 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 86:
#line 251 "parser.ypp"
                      { (yyval.str) = (yyvsp[0].str); }
#line 2595 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 87:
#line 254 "parser.ypp"
                                      { (yyval.d) = (yyvsp[0].d); }
#line 2601 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 88:
#line 255 "parser.ypp"
                                      { (yyval.d) = -(yyvsp[0].d); }
#line 2607 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 89:
#line 256 "parser.ypp"
                                      { (yyval.d) = std::numeric_limits<double>::infinity(); }
#line 2613 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 90:
#line 257 "parser.ypp"
                                      { (yyval.d) = -std::numeric_limits<double>::infinity();}
#line 2619 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 91:
#line 260 "parser.ypp"
                          { (yyval.d) = (yyvsp[0].i);  }
#line 2625 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 92:
#line 261 "parser.ypp"
                          { (yyval.d) = -(yyvsp[0].i); }
#line 2631 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 93:
#line 268 "parser.ypp"
                                             { }
#line 2637 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 94:
#line 271 "parser.ypp"
                      { }
#line 2643 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 95:
#line 272 "parser.ypp"
                      { }
#line 2649 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 96:
#line 275 "parser.ypp"
                              { }
#line 2655 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 97:
#line 276 "parser.ypp"
                              { }
#line 2661 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 98:
#line 279 "parser.ypp"
                                                    { rddlTask->addCPF(*(yyvsp[-3].parametrizedVariable), (yyvsp[-1].logicalExpression)); }
#line 2667 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 99:
#line 282 "parser.ypp"
                                                    {
                                                      std::string varName;
                                                      if ((*(yyvsp[0].str))[(yyvsp[0].str)->length() - 1] == '\'')
                                                        varName = (yyvsp[0].str)->substr(0, (yyvsp[0].str)->length() - 1);
                                                      else
                                                        varName = *(yyvsp[0].str);

                                                      if (rddlTask->variableDefinitions.find(varName) != rddlTask->variableDefinitions.end()) {
                                                        (yyval.parametrizedVariable) = rddlTask->variableDefinitions[varName];
                                                      } else {
                                                        SystemUtils::abort("Unknown parametrized variable " + varName + ".");
                                                      }
                                                      (yyval.parametrizedVariable) = new ParametrizedVariable(*(rddlTask->variableDefinitions[varName]), std::vector<Parameter*>());
                                                    }
#line 2686 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 100:
#line 296 "parser.ypp"
                                                    {
                                                      std::string varName;
                                                      if ((*(yyvsp[-3].str))[(yyvsp[-3].str)->length() - 1] == '\'')
                                                        varName = (yyvsp[-3].str)->substr(0, (yyvsp[-3].str)->length() - 1);
                                                      else
                                                        varName = *(yyvsp[-3].str);
                                                      if (rddlTask->variableDefinitions.find(varName) != rddlTask->variableDefinitions.end()) {
                                                        (yyval.parametrizedVariable) = rddlTask->variableDefinitions[varName];
                                                      } else {
                                                        SystemUtils::abort("Unknown parametrized variable " + varName + ".");
                                                      }
                                                      (yyval.parametrizedVariable) = new ParametrizedVariable(*(rddlTask->variableDefinitions[varName]), *(yyvsp[-1].parameters));
                                                    }
#line 2704 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 101:
#line 309 "parser.ypp"
                                                    { SystemUtils::abort("Undefined variable: " + *(yyvsp[-1].str) + ". Syntax not implemented yet.");     }
#line 2710 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 102:
#line 310 "parser.ypp"
                                                    { SystemUtils::abort("Undefined variable: " + *(yyvsp[-4].str) + ". Syntax not implemented yet.");     }
#line 2716 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 103:
#line 311 "parser.ypp"
                                                    { SystemUtils::abort("Undefined variable: " + *(yyvsp[-2].str) + ". Syntax not implemented yet.");     }
#line 2722 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 104:
#line 314 "parser.ypp"
                               { (yyval.parameters) = new std::vector<Parameter*>(); (yyval.parameters)->push_back((yyvsp[0].parameter)); }
#line 2728 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 105:
#line 315 "parser.ypp"
                               { (yyval.parameters) = (yyvsp[0].parameters); (yyval.parameters)->insert((yyval.parameters)->begin(), (yyvsp[-2].parameter)); }
#line 2734 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 106:
#line 318 "parser.ypp"
                             { (yyval.parameter) = new Parameter(*(yyvsp[0].str)); }
#line 2740 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 107:
#line 319 "parser.ypp"
                             { SystemUtils::abort("Definition of Term using parametrized variable not implemented yet."); }
#line 2746 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 108:
#line 320 "parser.ypp"
                             { assert(rddlTask->objects.find(*(yyvsp[0].str)) != rddlTask->objects.end()); (yyval.parameter) = rddlTask->objects[*(yyvsp[0].str)]; }
#line 2752 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 109:
#line 321 "parser.ypp"
                             { SystemUtils::abort("Definition of Term using variable with '$' sign not implemented yet."); }
#line 2758 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 110:
#line 324 "parser.ypp"
                               { SystemUtils::abort("MemberList not implemented yet."); }
#line 2764 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 111:
#line 325 "parser.ypp"
                               { SystemUtils::abort("MemberList not implemented yet."); }
#line 2770 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 112:
#line 328 "parser.ypp"
                                 { SystemUtils::abort("Pterm not implemented yet."); }
#line 2776 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 113:
#line 329 "parser.ypp"
                                 { SystemUtils::abort("Pterm not implemented yet."); }
#line 2782 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 114:
#line 330 "parser.ypp"
                                 { SystemUtils::abort("Pterm not implemented yet."); }
#line 2788 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 115:
#line 331 "parser.ypp"
                                 { SystemUtils::abort("Pterm not implemented yet."); }
#line 2794 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 116:
#line 332 "parser.ypp"
                                 { SystemUtils::abort("Pterm not implemented yet."); }
#line 2800 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 117:
#line 340 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = new Parameter(*(yyvsp[0].str)); }
#line 2806 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 118:
#line 341 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = (yyvsp[0].parametrizedVariable); }
#line 2812 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 119:
#line 342 "parser.ypp"
                                                                                                { assert(rddlTask->objects.find(*(yyvsp[0].str)) != rddlTask->objects.end()); (yyval.logicalExpression) = rddlTask->objects[*(yyvsp[0].str)]; }
#line 2818 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 120:
#line 343 "parser.ypp"
                                                                                                { SystemUtils::abort("'$<variable_name>' not implemented yet."); }
#line 2824 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 121:
#line 344 "parser.ypp"
                                                                                                { SystemUtils::abort("Structured expression list not implemented yet."); }
#line 2830 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 122:
#line 346 "parser.ypp"
                                                                                                { SystemUtils::abort("Unknow special function " + *(yyvsp[-3].str) + " defined as expression. Special functions not implemented yet."); }
#line 2836 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 123:
#line 347 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = new UniversalQuantification((yyvsp[-2].parameterList), (yyvsp[0].logicalExpression)); }
#line 2842 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 124:
#line 348 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = new ExistentialQuantification((yyvsp[-2].parameterList), (yyvsp[0].logicalExpression));}
#line 2848 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 125:
#line 349 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = new Sumation((yyvsp[-2].parameterList), (yyvsp[0].logicalExpression)); }
#line 2854 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 126:
#line 350 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = new Product((yyvsp[-2].parameterList), (yyvsp[0].logicalExpression)); }
#line 2860 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 127:
#line 351 "parser.ypp"
                                                                                                { (yyval.logicalExpression) = new IfThenElseExpression((yyvsp[-5].logicalExpression), (yyvsp[-2].logicalExpression), (yyvsp[0].logicalExpression)); }
#line 2866 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 128:
#line 352 "parser.ypp"
                                                                                                { // TODO: Text switch case control
                                                                                                   LogicalExpression* switchVar = rddlTask->getParametrizedVariable((yyvsp[-4].parameter)->name);

                                                                                                    std::vector<LogicalExpression*> conditions;
                                                                                                    std::vector<LogicalExpression*> effects;
                                                                                                    for (ConditionEffectPair* cs : *(yyvsp[-1].conditionEffects)) {
                                                                                                        if (!cs->first) {
                                                                                                            // If we reached the end of switch case and instead of 'case', we have 'default' -> default action
                                                                                                            conditions.push_back(new NumericConstant(1.0));
                                                                                                        }
                                                                                                        else {
                                                                                                            std::vector<LogicalExpression*> switchVarEquality;
                                                                                                            switchVarEquality.push_back(switchVar);
                                                                                                            switchVarEquality.push_back(cs->first);
                                                                                                            conditions.push_back(new EqualsExpression(switchVarEquality));
                                                                                                        }
                                                                                                        effects.push_back(cs->second);
                                                                                                    }
                                                                                                    (yyval.logicalExpression) = new MultiConditionChecker(conditions, effects);
                                                                                                }
#line 2891 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 129:
#line 373 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Addition(exprs); }
#line 2897 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 130:
#line 374 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back(new NumericConstant(0.0)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Subtraction(exprs); }
#line 2903 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 131:
#line 375 "parser.ypp"
                                                       { (yyval.logicalExpression) = (yyvsp[0].logicalExpression); }
#line 2909 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 132:
#line 376 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Subtraction(exprs); }
#line 2915 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 133:
#line 377 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Multiplication(exprs); }
#line 2921 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 134:
#line 378 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Division(exprs); }
#line 2927 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 135:
#line 379 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new GreaterExpression(exprs); }
#line 2933 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 136:
#line 380 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new LowerExpression(exprs); }
#line 2939 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 137:
#line 381 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new LowerEqualsExpression(exprs); }
#line 2945 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 138:
#line 382 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new GreaterEqualsExpression(exprs); }
#line 2951 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 139:
#line 383 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new EqualsExpression(exprs); }
#line 2957 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 140:
#line 384 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Negation(new EqualsExpression(exprs)); }
#line 2963 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 141:
#line 385 "parser.ypp"
                                                       { (yyval.logicalExpression) = (yyvsp[-1].logicalExpression); }
#line 2969 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 142:
#line 386 "parser.ypp"
                                                       { (yyval.logicalExpression) = (yyvsp[-1].logicalExpression); }
#line 2975 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 143:
#line 387 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Conjunction(exprs); }
#line 2981 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 144:
#line 388 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Conjunction(exprs); }
#line 2987 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 145:
#line 389 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back((yyvsp[-2].logicalExpression)); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Disjunction(exprs); }
#line 2993 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 146:
#line 390 "parser.ypp"
                                                       { (yyval.logicalExpression) = new Negation((yyvsp[0].logicalExpression)); }
#line 2999 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 147:
#line 391 "parser.ypp"
                                                       { std::vector<LogicalExpression*> posExprs, negExprs, exprs; posExprs.push_back((yyvsp[-2].logicalExpression)); negExprs.push_back((yyvsp[0].logicalExpression)); exprs.push_back(new Conjunction(posExprs)); exprs.push_back(new Conjunction(posExprs)); (yyval.logicalExpression) = new Disjunction(exprs); }
#line 3005 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 148:
#line 392 "parser.ypp"
                                                       { std::vector<LogicalExpression*> exprs; exprs.push_back(new Negation((yyvsp[-2].logicalExpression))); exprs.push_back((yyvsp[0].logicalExpression)); (yyval.logicalExpression) = new Disjunction(exprs); }
#line 3011 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 149:
#line 393 "parser.ypp"
                                                       { (yyval.logicalExpression) = new NumericConstant((yyvsp[0].d)); }
#line 3017 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 150:
#line 394 "parser.ypp"
                                                       { (yyval.logicalExpression) = new NumericConstant((yyvsp[0].i)); }
#line 3023 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 151:
#line 395 "parser.ypp"
                                                       { (yyval.logicalExpression) = new NumericConstant(1.0); }
#line 3029 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 152:
#line 396 "parser.ypp"
                                                       { (yyval.logicalExpression) = new NumericConstant(0.0); }
#line 3035 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 153:
#line 397 "parser.ypp"
                                                       { (yyval.logicalExpression) = new ExponentialFunction((yyvsp[-1].logicalExpression)); }
#line 3041 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 154:
#line 398 "parser.ypp"
                                                       { (yyval.logicalExpression) = new ExponentialFunction((yyvsp[-1].logicalExpression)); }
#line 3047 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 155:
#line 401 "parser.ypp"
                                                                            { (yyval.logicalExpression) = new BernoulliDistribution((yyvsp[-1].logicalExpression)); }
#line 3053 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 156:
#line 402 "parser.ypp"
                                                                            { (yyval.logicalExpression) = new KronDeltaDistribution((yyvsp[-1].logicalExpression)); }
#line 3059 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 157:
#line 403 "parser.ypp"
                                                                            { (yyval.logicalExpression) = (yyvsp[-1].lConstCaseList); }
#line 3065 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 158:
#line 404 "parser.ypp"
                                                                            { SystemUtils::abort("DiracDelta not implemtend."); }
#line 3071 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 159:
#line 405 "parser.ypp"
                                                                            { SystemUtils::abort("Uniform not implemtend."); }
#line 3077 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 160:
#line 406 "parser.ypp"
                                                                            { SystemUtils::abort("Normal not implemtend."); }
#line 3083 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 161:
#line 407 "parser.ypp"
                                                                            { SystemUtils::abort("Dirichelt not implemtend."); }
#line 3089 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 162:
#line 408 "parser.ypp"
                                                                            { SystemUtils::abort("Poisson not implemtend."); }
#line 3095 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 163:
#line 409 "parser.ypp"
                                                                            { SystemUtils::abort("Weibull not implemtend."); }
#line 3101 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 164:
#line 410 "parser.ypp"
                                                                            { SystemUtils::abort("Gama not implemtend.");  }
#line 3107 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 165:
#line 411 "parser.ypp"
                                                                            { SystemUtils::abort("Multinomial not implemtend."); }
#line 3113 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 166:
#line 414 "parser.ypp"
                                                                         { SystemUtils::abort("StructExpressionList not implemented yet."); }
#line 3119 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 167:
#line 415 "parser.ypp"
                                                                         { SystemUtils::abort("StructExpressionList not implemented yet."); }
#line 3125 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 168:
#line 418 "parser.ypp"
                                              { (yyval.logicalExpressions) = new std::vector<LogicalExpression*>(); (yyval.logicalExpressions)->push_back((yyvsp[0].logicalExpression)); }
#line 3131 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 169:
#line 419 "parser.ypp"
                                              { (yyval.logicalExpressions) = (yyvsp[0].logicalExpressions); (yyval.logicalExpressions)->insert((yyval.logicalExpressions)->begin(), (yyvsp[-2].logicalExpression)); }
#line 3137 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 170:
#line 422 "parser.ypp"
                                               { (yyval.parameterList) = new ParameterList({}, {}); (yyval.parameterList)->params.push_back(new Parameter((yyvsp[0].parameter)->name, (yyvsp[0].parameter)->type)); (yyval.parameterList)->types.push_back((yyvsp[0].parameter)->type); }
#line 3143 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 171:
#line 423 "parser.ypp"
                                               { (yyval.parameterList) = (yyvsp[0].parameterList); (yyval.parameterList)->params.insert((yyval.parameterList)->params.begin(), new Parameter((yyvsp[-2].parameter)->name, (yyvsp[-2].parameter)->type)); (yyval.parameterList)->types.insert((yyval.parameterList)->types.begin(), (yyvsp[-2].parameter)->type); }
#line 3149 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 172:
#line 426 "parser.ypp"
                                           { if (rddlTask->getType(*(yyvsp[0].str))) {
                                     (yyval.parameter) = new Parameter(*(yyvsp[-2].str), rddlTask->getType(*(yyvsp[0].str)));
                                   }
                                   else {
                                        SystemUtils::abort("Type " + *(yyvsp[0].str) + " not defined");
                                   }
                                 }
#line 3161 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 173:
#line 435 "parser.ypp"
                                        { (yyval.conditionEffects) = new std::vector<ConditionEffectPair*>(); (yyval.conditionEffects)->push_back((yyvsp[0].conditionEffect)); }
#line 3167 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 174:
#line 436 "parser.ypp"
                                        { (yyval.conditionEffects) = (yyvsp[0].conditionEffects); (yyval.conditionEffects)->insert((yyval.conditionEffects)->begin(), (yyvsp[-2].conditionEffect)); }
#line 3173 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 175:
#line 439 "parser.ypp"
                                                   { LogicalExpression* var = rddlTask->getParametrizedVariable((yyvsp[-2].parameter)->name); (yyval.conditionEffect) = new ConditionEffectPair(var, (yyvsp[0].logicalExpression)); }
#line 3179 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 176:
#line 440 "parser.ypp"
                                                   { (yyval.conditionEffect) = new ConditionEffectPair(nullptr, (yyvsp[0].logicalExpression)); }
#line 3185 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 177:
#line 444 "parser.ypp"
                                                          { (yyval.lConstCaseList) = new DiscreteDistribution(); (yyval.lConstCaseList)->values.push_back((yyvsp[-2].logicalExpression)); (yyval.lConstCaseList)->probabilities.push_back((yyvsp[0].logicalExpression)); }
#line 3191 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 178:
#line 445 "parser.ypp"
                                                          { SystemUtils::abort("Key word 'otherwise' not supported yet"); }
#line 3197 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 179:
#line 446 "parser.ypp"
                                                          { (yyval.lConstCaseList) = (yyvsp[0].lConstCaseList); (yyval.lConstCaseList)->values.insert((yyval.lConstCaseList)->values.begin(), (yyvsp[-4].logicalExpression)); (yyval.lConstCaseList)->probabilities.insert((yyval.lConstCaseList)->probabilities.begin(), (yyvsp[-2].logicalExpression)); }
#line 3203 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 180:
#line 452 "parser.ypp"
                                                       { rddlTask->setRewardCPF((yyvsp[-1].logicalExpression)); }
#line 3209 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 181:
#line 459 "parser.ypp"
                                                                                               { }
#line 3215 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 182:
#line 460 "parser.ypp"
                                                                                               { }
#line 3221 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 183:
#line 463 "parser.ypp"
                                                                     { }
#line 3227 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 184:
#line 464 "parser.ypp"
                                                                     { }
#line 3233 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 185:
#line 467 "parser.ypp"
                                         { rddlTask->SACs.push_back((yyvsp[-1].logicalExpression)); }
#line 3239 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 186:
#line 473 "parser.ypp"
                                                                                            { }
#line 3245 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 187:
#line 474 "parser.ypp"
                                                                                            { }
#line 3251 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 188:
#line 477 "parser.ypp"
                                                                            { }
#line 3257 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 189:
#line 478 "parser.ypp"
                                                                            { }
#line 3263 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 190:
#line 481 "parser.ypp"
                                             { rddlTask->SACs.push_back((yyvsp[-1].logicalExpression)); }
#line 3269 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 191:
#line 487 "parser.ypp"
                                                                             { }
#line 3275 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 192:
#line 488 "parser.ypp"
                                                                             { }
#line 3281 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 196:
#line 501 "parser.ypp"
                                                      { }
#line 3287 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 197:
#line 505 "parser.ypp"
                                          { }
#line 3293 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 198:
#line 506 "parser.ypp"
                                          { }
#line 3299 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 199:
#line 509 "parser.ypp"
                                                            {
                                                              for (std::string str : *(yyvsp[-2].strs)) {
                                                                rddlTask->addObject(*(yyvsp[-5].str), str);
                                                              }
                                                         }
#line 3309 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 200:
#line 516 "parser.ypp"
                                                  { (yyval.strs) = new std::vector<std::string>(); (yyval.strs)->push_back(*(yyvsp[0].str)); }
#line 3315 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 201:
#line 517 "parser.ypp"
                                                  { (yyval.strs) = (yyvsp[0].strs); (yyval.strs)->insert((yyval.strs)->begin(), *(yyvsp[-2].str)); }
#line 3321 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 202:
#line 518 "parser.ypp"
                                                  { SystemUtils::abort("Definition of an object using '$' not implemented yet."); }
#line 3327 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 203:
#line 519 "parser.ypp"
                                                  { SystemUtils::abort("Definition of an object using '$' not implemented yet."); }
#line 3333 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 204:
#line 531 "parser.ypp"
                {
                    rddlTask->nonFluentsName = *(yyvsp[-16].str);
                    if (rddlTask->domainName != *(yyvsp[-12].str)) {
                        SystemUtils::abort("Unknown domain " + *(yyvsp[-12].str) +
                                           "  used in non-fluents section");
                    }
                }
#line 3345 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 205:
#line 541 "parser.ypp"
                {
                    rddlTask->nonFluentsName = *(yyvsp[-11].str);
                    if (rddlTask->domainName != *(yyvsp[-7].str)) {
                        SystemUtils::abort("Unknown domain " + *(yyvsp[-7].str) +
                                           "  used in non-fluents section");
                    }
                }
#line 3357 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 206:
#line 551 "parser.ypp"
                {
                    rddlTask->nonFluentsName = *(yyvsp[-11].str);
                    if (rddlTask->domainName != *(yyvsp[-7].str)) {
                        SystemUtils::abort("Unknown domain " + *(yyvsp[-7].str) +
                                           "  used in non-fluents section");
                    }
                }
#line 3369 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 207:
#line 560 "parser.ypp"
                                                                        { }
#line 3375 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 208:
#line 561 "parser.ypp"
                                                                        { }
#line 3381 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 209:
#line 565 "parser.ypp"
                                                                              {
                                                                                ParametrizedVariable* parent = rddlTask->getParametrizedVariable(*(yyvsp[-4].str));
                                                                                std::vector<Parameter*> params;
                                                                                for (LogicalExpression* le : *(yyvsp[-2].logicalExpressions)) {
                                                                                    Parameter* param = dynamic_cast<Parameter*>(le);
                                                                                    if (!param) {
                                                                                        SystemUtils::abort("Passing anything other than Parameter object to ParametrizedVariable is not allowed.");
                                                                                    }
                                                                                    params.push_back(rddlTask->getObject(param->name));
                                                                                }
                                                                                rddlTask->addParametrizedVariable(parent, params, 1);
                                                                              }
#line 3398 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 210:
#line 577 "parser.ypp"
                                                                              { ParametrizedVariable* parent = rddlTask->getParametrizedVariable(*(yyvsp[-1].str)); rddlTask->addParametrizedVariable(parent,  {}, 1); }
#line 3404 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 211:
#line 578 "parser.ypp"
                                                                              {
                                                                                ParametrizedVariable* parent = rddlTask->getParametrizedVariable(*(yyvsp[-4].str));
                                                                                std::vector<Parameter*> params;
                                                                                for (LogicalExpression* le : *(yyvsp[-2].logicalExpressions)) {
                                                                                    Parameter* param = dynamic_cast<Parameter*>(le);
                                                                                    if (!param) {
                                                                                        SystemUtils::abort("Passing anything other than Parameter object to ParametrizedVariable is not allowed.");
                                                                                    }
                                                                                    params.push_back(rddlTask->getObject(param->name));
                                                                                }
                                                                                rddlTask->addParametrizedVariable(parent, params, 0);
                                                                              }
#line 3421 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 212:
#line 590 "parser.ypp"
                                                                              { ParametrizedVariable* parent = rddlTask->getParametrizedVariable(*(yyvsp[0].str)); rddlTask->addParametrizedVariable(parent,  {}, 0); }
#line 3427 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 213:
#line 591 "parser.ypp"
                                                                              {
                                                                                ParametrizedVariable* parent = rddlTask->getParametrizedVariable(*(yyvsp[-6].str));
                                                                                std::vector<Parameter*> params;
                                                                                for (LogicalExpression* le : *(yyvsp[-4].logicalExpressions)) {
                                                                                    Parameter* param = dynamic_cast<Parameter*>(le);
                                                                                    if (!param) {
                                                                                        SystemUtils::abort("Passing anything other than Parameter object to ParametrizedVariable is not allowed.");
                                                                                    }
                                                                                    params.push_back(rddlTask->getObject(param->name));
                                                                                }
                                                                                rddlTask->addParametrizedVariable(parent, params, (yyvsp[-1].d));
                                                                              }
#line 3444 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 214:
#line 603 "parser.ypp"
                                                                              { ParametrizedVariable* parent = rddlTask->getParametrizedVariable(*(yyvsp[-3].str)); rddlTask->addParametrizedVariable(parent,  {}, (yyvsp[-1].d));}
#line 3450 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 215:
#line 606 "parser.ypp"
                                    { (yyval.logicalExpressions) = new std::vector<LogicalExpression*>(); (yyval.logicalExpressions)->push_back((yyvsp[0].logicalExpression)); }
#line 3456 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 216:
#line 607 "parser.ypp"
                                    { (yyval.logicalExpressions) = (yyvsp[0].logicalExpressions); (yyval.logicalExpressions)->insert((yyval.logicalExpressions)->begin(), (yyvsp[-2].logicalExpression)); }
#line 3462 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 217:
#line 625 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-30].str), *(yyvsp[-26].str), *(yyvsp[-22].str), (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3468 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 218:
#line 634 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-26].str), *(yyvsp[-22].str), "", (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3474 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 219:
#line 643 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-25].str), *(yyvsp[-21].str), *(yyvsp[-17].str), (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3480 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 220:
#line 651 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-21].str), *(yyvsp[-17].str), "", (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3486 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 221:
#line 660 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-25].str), *(yyvsp[-21].str), *(yyvsp[-17].str), (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3492 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 222:
#line 668 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-21].str), *(yyvsp[-17].str), "", (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3498 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 223:
#line 676 "parser.ypp"
                                  {  rddlTask->setInstance(*(yyvsp[-20].str), *(yyvsp[-16].str), *(yyvsp[-12].str), (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d));  }
#line 3504 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 224:
#line 683 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-16].str), *(yyvsp[-12].str), "", (yyvsp[-8].i), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3510 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 225:
#line 694 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-26].str), *(yyvsp[-22].str), *(yyvsp[-18].str), std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3516 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 226:
#line 702 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-22].str), *(yyvsp[-18].str), "", std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3522 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 227:
#line 710 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-21].str), *(yyvsp[-17].str), *(yyvsp[-13].str), std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3528 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 228:
#line 717 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-17].str), *(yyvsp[-13].str), "", std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3534 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 229:
#line 725 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-21].str), *(yyvsp[-17].str), *(yyvsp[-13].str), std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3540 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 230:
#line 732 "parser.ypp"
                                  { rddlTask->setInstance(*(yyvsp[-17].str), *(yyvsp[-13].str), "", std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3546 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 231:
#line 739 "parser.ypp"
                                   { rddlTask->setInstance(*(yyvsp[-16].str), *(yyvsp[-12].str), *(yyvsp[-8].str), std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3552 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 232:
#line 745 "parser.ypp"
                                   { rddlTask->setInstance(*(yyvsp[-12].str), *(yyvsp[-8].str), "", std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3558 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 233:
#line 754 "parser.ypp"
                                   { rddlTask->setInstance(*(yyvsp[-27].str), *(yyvsp[-23].str), "", std::numeric_limits<int>::max(), (yyvsp[-6].i), (yyvsp[-2].d)); }
#line 3564 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 234:
#line 758 "parser.ypp"
                                                                                { (yyval.i) = (yyvsp[0].i); }
#line 3570 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 235:
#line 759 "parser.ypp"
                                                                                { SystemUtils::abort("Definition of horizon using 'terminate-when' token not implemented yet."); }
#line 3576 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 236:
#line 762 "parser.ypp"
                                                           { (yyval.i) = (yyvsp[0].i); }
#line 3582 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;

  case 237:
#line 763 "parser.ypp"
                                                          { (yyval.i) = std::numeric_limits<int>::max(); }
#line 3588 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"
    break;


#line 3592 "/home/ziqi/prost-master/builds/debug/rddl_parser/parser.tab.cc"

      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */
  {
    const int yylhs = yyr1[yyn] - YYNTOKENS;
    const int yyi = yypgoto[yylhs] + *yyssp;
    yystate = (0 <= yyi && yyi <= YYLAST && yycheck[yyi] == *yyssp
               ? yytable[yyi]
               : yydefgoto[yylhs]);
  }

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = YY_CAST (char *, YYSTACK_ALLOC (YY_CAST (YYSIZE_T, yymsg_alloc)));
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:
  /* Pacify compilers when the user code never invokes YYERROR and the
     label yyerrorlab therefore never appears in user code.  */
  if (0)
    YYERROR;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;


/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;


#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif


/*-----------------------------------------------------.
| yyreturn -- parsing is finished, return the result.  |
`-----------------------------------------------------*/
yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[+*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 766 "parser.ypp"


bool checkExtension(std::string s) {
     return ((s.length() > 5) &&  (s.substr(s.length() - 5).compare(".rddl") == 0));
}

int main (int argc, char** argv) {


    Timer t;
    std::cout << "Parsing..." << std::endl;
    if (argc < 3) {
        // Run unit tests
        doctest::Context context;
        context.applyCommandLine(argc, argv);
        int res = context.run();

        if (context.shouldExit()) {
            return res;
        }

        SystemUtils::abort("Usage: ./rddl-parser <rddlDesc> <targetDir-or-targetFile> [options]\n"
                           "where rddlDesc consists of 1-3 individual files");
    }

    // Find input files and combine them in one file
    std::stringstream combined;
    unsigned int index = 1;

    while (index < argc && checkExtension(argv[index])) {
         std::ifstream ifs(argv[index], std::ifstream::in);
         combined << ifs.rdbuf();
         ifs.close();
         index++;
    }
    if (index == 1 || index > 4 || index >= argc) {
        SystemUtils::abort("Usage: ./rddl-parser <rddlDesc> <targetDir> [options]\n"
                           "where rddlDesc consists of 1-3 individual files");
    }

    std::string targetDir = std::string(argv[index++]);

    double seed = time(nullptr);
    int numStates = 250;
    int numSimulations = 25;
    double timeout = 10.0;
    bool useIPC2018Rules = false;

    // Read optionals
    for (; index < argc; ++index) {
        std::string nextOption = std::string(argv[index]);
        if (nextOption == "-s") {
            seed = atoi(std::string(argv[++index]).c_str());
            std::cout << "Setting seed to " << seed << std::endl;
        } else if (nextOption == "-trainingSimulations") {
            numSimulations = atoi(std::string(argv[++index]).c_str());
            std::cout << "Setting number of simulations for training set creation to "
                      << numSimulations << std::endl;
        } else if (nextOption == "-trainingSetSize") {
            numStates = atoi(std::string(argv[++index]).c_str());
            std::cout << "Setting target training set size to " << numStates << std::endl;
        } else if (nextOption == "-trainingTimeout") {
            timeout = atof(std::string(argv[++index]).c_str());
            std::cout << "Setting training timeout to " << timeout << std::endl;
        } else if (nextOption == "-ipc2018") {
            useIPC2018Rules = atoi(std::string(argv[++index]).c_str());
            std::cout << "Using IPC 2018 rules: " << useIPC2018Rules << std::endl;
        } else {
            assert(false);
        }
    }

    // Creating RDDLTask object
    rddlTask = new RDDLTask();

    yy_scan_string(combined.str().c_str());
    yyparse();
    std::cout << "...finished (" << t << ")." << std::endl;

    rddlTask->execute(targetDir, seed, numStates, numSimulations, timeout, useIPC2018Rules);
    std::cout << "PROST parser complete running time: " << t << std::endl;

    return EXIT_SUCCESS;
}
